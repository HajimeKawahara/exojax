"""Compatibility layer for the deprecated :mod:`exojax.spec` namespace.

The implementation moved to :mod:`exojax.opacity`, :mod:`exojax.rt`,
:mod:`exojax.database`, and :mod:`exojax.postproc`.  The old module paths are
kept importable until ExoJAX v3.0 so that both of these forms continue to work::

    from exojax.spec import lpf
    from exojax.spec.lpf import xsvector

New code should import from the new namespaces directly.
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from importlib.machinery import ModuleSpec
from typing import Final


warnings.warn(
    "`exojax.spec` is deprecated and will be removed in v3.0. "
    "Switch to `exojax.opacity`, `exojax.rt`, `exojax.database`, or "
    "`exojax.postproc`.",
    DeprecationWarning,
    stacklevel=2,
)


# A module can have more than one target when an old, monolithic module was
# split.  Attribute lookup searches the targets in order without importing any
# of them merely because ``exojax.spec`` itself was imported.
_MODULE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    # Opacity calculators and kernels
    "opacalc": ("exojax.opacity.opacalc",),
    "initspec": ("exojax.opacity.initspec",),
    "premodit": ("exojax.opacity.premodit.premodit",),
    "modit": ("exojax.opacity.modit.modit",),
    "lpf": ("exojax.opacity.lpf.lpf",),
    "set_ditgrid": ("exojax.opacity._common.set_ditgrid",),
    "optgrid": ("exojax.opacity.premodit.optgrid",),
    "lbd": ("exojax.opacity.premodit.lbd",),
    "lsd": ("exojax.opacity._common.lsd",),
    "lbderror": ("exojax.opacity.premodit.lbderror",),
    "dit": ("exojax.opacity.modit.dit",),
    "ditkernel": ("exojax.opacity._common.ditkernel",),
    "make_numatrix": ("exojax.opacity.lpf.make_numatrix",),
    "opacont": ("exojax.opacity.opacont",),
    "rayleigh": ("exojax.opacity.rayleigh",),
    "generate_elower_grid_trange": (
        "exojax.opacity.premodit.generate_elower_grid_trange",
    ),
    "profconv": ("exojax.opacity._common.profconv",),
    "lpffilter": ("exojax.opacity._common.lpffilter",),
    # Radiative transfer
    "chord": ("exojax.rt.chord",),
    "atmrt": ("exojax.rt.atmrt",),
    "opart": ("exojax.rt.emis", "exojax.rt.reflect"),
    "layeropacity": ("exojax.rt.layeropacity",),
    "planck": ("exojax.rt.planck",),
    "rtlayer": ("exojax.rt.rtlayer",),
    "rtransfer": ("exojax.rt.rtransfer",),
    "toon": ("exojax.rt.toon",),
    "twostream": ("exojax.rt.twostream",),
    # Databases.  Several v2.0 modules were subsequently split into packages.
    "mie": ("exojax.database.mie",),
    "api": ("exojax.database.api", "exojax.database._common.commonapi"),
    "atomll": (
        "exojax.database.core_atom.broadening",
        "exojax.database.core_atom.line_strength",
        "exojax.database.core_atom.misc",
        "exojax.database.core_atom.pf",
    ),
    # v2.2--v2.5 accidentally advertised the malformed key ``core_atom.io``;
    # ``atomllapi`` is the real v2 module path retained here.
    "atomllapi": ("exojax.database.core_atom.io", "exojax.database.core_atom.pf"),
    "contdb": ("exojax.database.contdb",),
    "customapi": ("exojax.database.hargreaves.api",),
    # The old dbmanager base classes were deliberately removed with their
    # download feature.  The Hargreaves API is the closest supported endpoint.
    "dbmanager": ("exojax.database.hargreaves.api",),
    "exomol": ("exojax.database.core.broadening",),
    "exomolhr": ("exojax.database.exomolhr.api",),
    "hitran": (
        "exojax.database.core.line_strength",
        "exojax.database.core.broadening",
    ),
    "hitranapi": ("exojax.database._common.hitranapi",),
    "hitrancia": (
        "exojax.database.cia.io",
        "exojax.database.core.abscoeff",
    ),
    "hminus": ("exojax.database.hminus",),
    "moldb": ("exojax.database",),
    "molinfo": ("exojax.database.molinfo",),
    "multimol": ("exojax.database.multimol",),
    "nonair": ("exojax.database.nonair",),
    "pardb": ("exojax.database.pardb",),
    "qstate": ("exojax.database.molinfo.qstate",),
    # Post-processing
    "limb_darkening": ("exojax.postproc.limb_darkening",),
    "response": ("exojax.postproc.response",),
    "specop": ("exojax.postproc.specop",),
    "spin_rotation": ("exojax.postproc.spin_rotation",),
}


_OBJECT_ALIASES: Final[dict[str, tuple[str, str]]] = {
    "OpaPremodit": ("exojax.opacity", "OpaPremodit"),
    "OpaDirect": ("exojax.opacity", "OpaDirect"),
    "OpaModit": ("exojax.opacity", "OpaModit"),
    # Names historically re-exported by ``exojax.spec`` itself.
    "line_strength": ("exojax.database.core.line_strength", "line_strength"),
    "doppler_sigma": ("exojax.database.core.broadening", "doppler_sigma"),
    "gamma_natural": ("exojax.database.core.broadening", "gamma_natural"),
    "normalized_doppler_sigma": (
        "exojax.database.core.broadening",
        "normalized_doppler_sigma",
    ),
    "hjert": ("exojax.opacity.lpf.lpf", "hjert"),
    "voigt": ("exojax.opacity.lpf.lpf", "voigt"),
    "voigtone": ("exojax.opacity.lpf.lpf", "voigtone"),
    "vvoigt": ("exojax.opacity.lpf.lpf", "vvoigt"),
    "make_numatrix0": ("exojax.opacity.lpf.make_numatrix", "make_numatrix0"),
}


# Historical introspection/migration helpers imported this private table.
_ALIASES: Final[dict[str, str]] = {
    **{name: targets[0] for name, targets in _MODULE_ALIASES.items()},
    **{
        name: f"{module_name}.{attribute_name}"
        for name, (module_name, attribute_name) in _OBJECT_ALIASES.items()
    },
}


# Retained for the bundled ``scripts/fix_v2_1.py`` import-rewrite utility.
_SUBMODULES: Final[dict[str, str]] = {
    "opacalc": "exojax.opacity",
    "lpf": "exojax.opacity.lpf",
    "modit": "exojax.opacity.modit",
    "premodit": "exojax.opacity.premodit",
}


# Names that changed while the implementation was being split.  This keeps the
# most commonly used v2.0 symbol spelling working as well as its module path.
_RENAMED_SYMBOLS: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "atomll": {
        "Sij0": (
            "exojax.database.core_atom.line_strength",
            "line_strength_atom",
        ),
    },
    "customapi": {
        "set_wavenum": (
            "exojax.database.hargreaves.api",
            "_set_wavenum_hargreaves",
        ),
    },
    "exomolhr": {
        "fetch_opacity_zip": (
            "exojax.provider.exomolhr",
            "_fetch_opacity_zip",
        ),
        "load_exomolhr_csv": (
            "exojax.provider.exomolhr",
            "_load_exomolhr_csv",
        ),
        "list_isotopologues": (
            "exojax.database._common.isotope_functions",
            "_list_isotopologues",
        ),
    },
    "opacalc": {
        "OpaCalc": ("exojax.opacity.base", "OpaCalc"),
    },
    "specop": {
        "SopCommon": ("exojax.postproc.specop", "SopCommonConv"),
    },
}


class _MovedModule(types.ModuleType):
    """Lazy module proxy for an old :mod:`exojax.spec` module."""

    def __init__(self, old_fqn: str, targets: tuple[str, ...]):
        super().__init__(old_fqn)
        self.__package__ = old_fqn.rpartition(".")[0]
        self.__loader__ = None
        self.__spec__ = ModuleSpec(old_fqn, loader=None)
        self._targets = targets
        self.__doc__ = "Deprecated alias for " + ", ".join(
            f":mod:`{target}`" for target in targets
        )

    def __getattr__(self, name: str):
        old_name = self.__name__.removeprefix(f"{__name__}.")
        if name == "__all__":
            public_names = set(_RENAMED_SYMBOLS.get(old_name, {}))
            for target in self._targets:
                module = importlib.import_module(target)
                explicit = module.__dict__.get("__all__")
                if explicit is None:
                    public_names.update(
                        item for item in vars(module) if not item.startswith("_")
                    )
                else:
                    public_names.update(explicit)
            value = sorted(public_names)
            self.__all__ = value
            return value

        renamed = _RENAMED_SYMBOLS.get(old_name, {}).get(name)
        if renamed is not None:
            module_name, new_name = renamed
            value = getattr(importlib.import_module(module_name), new_name)
            setattr(self, name, value)
            return value

        for target in self._targets:
            module = importlib.import_module(target)
            try:
                value = getattr(module, name)
            except AttributeError:
                continue
            setattr(self, name, value)
            return value

        targets = ", ".join(repr(target) for target in self._targets)
        raise AttributeError(
            f"module {self.__name__!r} has no attribute {name!r}; "
            f"searched {targets}"
        )

    def __dir__(self):
        names = set(self.__dict__)
        old_name = self.__name__.removeprefix(f"{__name__}.")
        names.update(_RENAMED_SYMBOLS.get(old_name, {}))
        for target in self._targets:
            try:
                names.update(dir(importlib.import_module(target)))
            except ImportError:
                # Optional dependencies should not make tab completion fail.
                continue
        return sorted(names)


def _install_module_alias(old_name: str) -> _MovedModule:
    old_fqn = f"{__name__}.{old_name}"
    existing = sys.modules.get(old_fqn)
    if isinstance(existing, _MovedModule):
        return existing

    proxy = _MovedModule(old_fqn, _MODULE_ALIASES[old_name])
    sys.modules[old_fqn] = proxy
    globals()[old_name] = proxy
    return proxy


for _old_name in _MODULE_ALIASES:
    _install_module_alias(_old_name)


def __getattr__(name: str):
    """Resolve deprecated class names on first access."""
    target = _OBJECT_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(set(_MODULE_ALIASES) | set(_OBJECT_ALIASES))
