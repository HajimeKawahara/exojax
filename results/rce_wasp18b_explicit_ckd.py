"""Bind large CKD arrays at runtime without changing the frozen P7 equations.

This process-local adapter supports the frozen forward and validation callers,
including continuation and stellar-flux diagnostics. It changes compilation
inputs only. It is not an implicit-derivative adapter or a general JAX patch.
Use a fresh, single-threaded caller and keep the context open through execution.
"""

from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import threading

import jax


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "examples/_rce_wasp18b.py": "89bb891e0fe50824790ae933ee3f8989d3afa9b3018d48b9042fe8a8a22c1836",
    "examples/_rce_wasp18b_continuation.py": "2e196f7384cd82d632296d185f85574155d2d978b4cf40852ad05f85884837ff",
    "examples/rce_wasp18b_forward.py": "ad8e54928f806f436cba365468e3ebd061288df381cb88189a2533b691ff448c",
    "examples/rce_wasp18b_validation.py": "b02b51e39a6602810f2e0a3b357b7a8bea444984458b0e9dc4d97abfe5c80486",
    "examples/_rce_wasp18b_audit.py": "d010a0c03166323e441662640f712f269cf6e44e80fc52c0cf2d974b12f2507f",
    "src/exojax/atm/rce_device.py": "494dcc6fa86a0dad0ff48ca7639c642e198e3ae5bf3f51f1caae0fde323a7390",
    "src/exojax/opacity/ckd/api.py": "d7e85b4c8eb9e2ba155bcc4e8f969fbd0c781ecf781fb19c3e64c79c5a1dc448",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _BoundExecutable:
    def __init__(self, owner, executable):
        self.owner, self.executable = owner, executable

    def __call__(self, *args, **kwargs):
        self.owner.check_identity()
        return self.executable(self.owner.tables, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.executable, name)


class _BoundLowered(_BoundExecutable):
    def compile(self, *args, **kwargs):
        return _BoundExecutable(self.owner, self.executable.compile(*args, **kwargs))


class ExplicitCKDFunction:
    """Keep the caller signature while passing log_kggrid as JIT operands."""

    def __init__(self, function, column):
        self.function, self.column = function, column
        self.names = tuple(column.ckd)
        self.opacities = tuple(column.ckd.values())
        self.infos = tuple(opa.ckd_info for opa in self.opacities)
        self.tables = tuple(info.log_kggrid for info in self.infos)
        self.owner_thread = threading.get_ident()
        if not self.tables or len({id(opa) for opa in self.opacities}) != len(self.tables):
            raise ValueError("Require distinct nonempty CKD opacity objects.")
        self.compiled = jax.jit(self.execute)

    def check_identity(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("CKD binding requires its original caller thread.")
        if tuple(self.column.ckd) != self.names or any(
            self.column.ckd[name] is not opa or opa.ckd_info is not info
            or info.log_kggrid is not table
            for name, opa, info, table in zip(self.names, self.opacities, self.infos, self.tables)
        ):
            raise RuntimeError("CKD objects changed after explicit binding.")

    def execute(self, tables, *args, **kwargs):
        self.check_identity()
        if len(tables) != len(self.tables) or any(
            value.shape != original.shape or value.dtype != original.dtype
            for value, original in zip(tables, self.tables)
        ):
            raise ValueError("Explicit CKD operands must preserve shape and dtype.")
        try:
            for opa, info, table in zip(self.opacities, self.infos, tables):
                opa.ckd_info = replace(info, log_kggrid=table)
            return self.function(*args, **kwargs)
        finally:
            for opa, info in zip(self.opacities, self.infos):
                opa.ckd_info = info

    def __call__(self, *args, **kwargs):
        self.check_identity()
        return self.compiled(self.tables, *args, **kwargs)

    def lower(self, *args, **kwargs):
        self.check_identity()
        return _BoundLowered(self, self.compiled.lower(self.tables, *args, **kwargs))

    def description(self):
        return {"function": self.function.__qualname__, "tables": [
            {"species": name, "shape": list(table.shape), "dtype": str(table.dtype),
             "bytes": int(table.size * table.dtype.itemsize)}
            for name, table in zip(self.names, self.tables)
        ]}


class _LocalJax:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(jax, name)

    def jit(self, function, *args, **kwargs):
        column = getattr(function, "__self__", None)
        recognized = (isinstance(column, self.context.column_type)
                      and getattr(function, "__func__", None) in (
                          self.context.column_type.fluxes,
                          self.context.column_type.optical_depth))
        if isinstance(function, ExplicitCKDFunction) or recognized:
            if args or kwargs:
                raise ValueError("Frozen CKD call sites must use the default jit options.")
            return (function if isinstance(function, ExplicitCKDFunction)
                    else self.context.bind(function, column))
        return jax.jit(function, *args, **kwargs)


class ExplicitCKDContext:
    """Patch only the frozen callers' bindings; restore them on every exit."""

    _active = False

    def __init__(self):
        self.patches, self.bindings = [], []
        self.restored = False
        self.owner_thread = threading.get_ident()

    def _patch(self, module, name, replacement):
        self.patches.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def bind(self, function, column):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("CKD context does not support concurrent caller threads.")
        wrapped = ExplicitCKDFunction(function, column)
        # Keep metadata only: retaining bound functions would pin old suite tables.
        self.bindings.append(wrapped.description())
        return wrapped

    def identity(self):
        return {
            "path": str(Path(__file__).resolve()), "sha256": _sha(__file__),
            "scope": "Only CKD log_kggrid becomes an explicit JIT argument; frozen equations, precision, seeds and limits are unchanged.",
            "call_sites": ["column solver factory", "continuation solver factory",
                           "forward fluxes", "validation fluxes and optical_depth",
                           "stellar_flux_audit absorption_depth"],
            "source_sha256": dict(SOURCE_SHA256),
        }

    def evidence(self):
        return dict(self.identity(), bindings=list(self.bindings), restored=self.restored)

    def _restore(self):
        for module, name, original in reversed(self.patches):
            setattr(module, name, original)
        self.patches.clear()
        self.restored = True
        type(self)._active = False

    def __enter__(self):
        if type(self)._active or self.restored:
            raise RuntimeError("Use one fresh explicit CKD context at a time.")
        if jax.__version__ != "0.6.2":
            raise RuntimeError("This execution adapter was validated only with JAX 0.6.2.")
        for relative, expected in SOURCE_SHA256.items():
            if _sha(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen source changed: {relative}")
        type(self)._active = True
        try:
            column_module, continuation, forward, validation, audit = [
                importlib.import_module(name) for name in (
                    "_rce_wasp18b", "_rce_wasp18b_continuation", "rce_wasp18b_forward",
                    "rce_wasp18b_validation", "_rce_wasp18b_audit")]
            self.column_type = column_module.Wasp18bColumn
            original_factory = column_module.make_device_rce_solver
            if continuation.make_device_rce_solver is not original_factory:
                raise RuntimeError("The frozen solver factory bindings disagree.")

            def factory(*args, **kwargs):
                column = getattr(kwargs.get("valid_temperature"), "__self__", None)
                if not isinstance(column, self.column_type):
                    raise ValueError("Explicit CKD solver requires a bound P7 column guard.")
                original = original_factory(*args, **kwargs)
                return self.bind(original.__wrapped__, column)

            for module in (column_module, continuation):
                self._patch(module, "make_device_rce_solver", factory)
            for module in (forward, validation):
                if module.jax is not jax:
                    raise RuntimeError("Unexpected JAX binding before adapter installation.")
                self._patch(module, "jax", _LocalJax(self))
            original_audit = audit.stellar_flux_audit

            def stellar_flux_audit(column, temperature, parameters, chemistry, *, absorption_depth=None):
                if absorption_depth is None:
                    absorption_depth = self.bind(column.optical_depth, column)(temperature, chemistry)
                return original_audit(column, temperature, parameters, chemistry,
                                      absorption_depth=absorption_depth)

            self._patch(audit, "stellar_flux_audit", stellar_flux_audit)
            original_provenance = forward.provenance

            def provenance():
                # The report owns this metadata list before column preparation;
                # subsequent checkpoints serialize all bindings created so far.
                return dict(original_provenance(), execution_adapter=dict(
                    self.identity(), bindings=self.bindings))

            self._patch(forward, "provenance", provenance)
            self._patch(validation, "provenance", provenance)
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self._restore()
        changed = [relative for relative, expected in SOURCE_SHA256.items()
                   if _sha(ROOT / relative) != expected]
        if changed:
            raise RuntimeError(f"Frozen source changed during execution: {changed}")
        return False
