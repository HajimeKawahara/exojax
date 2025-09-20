"""Deprecated compatibility layer for the old `exojax.spec` namespace.

All symbols will be **removed in ExoJAX v3.0**.
Update your code to the new packages:

    exojax.opacity.*   (opacity calculators)
    exojax.rt.*        (radiative-transfer utilities)
    exojax.database.*  (line-list databases)
    exojax.postproc.*  (post-processing helpers)
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Final, Dict

# ────────────────────────────────────────────────────────────────────
# Emit the deprecation warning only once
# ────────────────────────────────────────────────────────────────────
warnings.warn(
    "`exojax.spec` is deprecated and will be removed in v3.0. "
    "Switch to `exojax.opacity`, `exojax.rt`, …",
    DeprecationWarning,
    stacklevel=2,
)

# ────────────────────────────────────────────────────────────────────
# 1) Symbol-level aliases  (old attr  → fully-qualified new attr)
# ────────────────────────────────────────────────────────────────────
_ALIASES: Final[Dict[str, str]] = {
    # --- main opacity APIs ---
    "opacalc": "exojax.opacity.opacalc",
    "initspec": "exojax.opacity.initspec",
    "premodit": "exojax.opacity.premodit.premodit",
    "modit": "exojax.opacity.modit.modit",
    "lpf": "exojax.opacity.lpf.lpf",
    "set_ditgrid": "exojax.opacity._common.set_ditgrid",
    "optgrid": "exojax.opacity.premodit.optgrid",
    "lbd": "exojax.opacity.premodit.lbd",
    "lsd": "exojax.opacity._common.lsd",
    "lbderror": "exojax.opacity.premodit.lbderror",
    "dit": "exojax.opacity.modit.dit",
    "ditkernel": "exojax.opacity._common.ditkernel",
    "make_numatrix": "exojax.opacity.lpf.make_numatrix",
    "opacont": "exojax.opacity.opacont",
    "rayleigh": "exojax.opacity.rayleigh",
    "generate_elower_grid_trange": "exojax.opacity.premodit.generate_elower_grid_trange",
    "profconv": "exojax.opacity._common.profconv",
    "lpffilter": "exojax.opacity._common.lpffilter",
    "chord": "exojax.rt.chord",
    "atmrt": "exojax.rt.atmrt",
    "opart": "exojax.rt.opart",
    "layeropacity": "exojax.rt.layeropacity",
    "planck": "exojax.rt.planck",
    "rtlayer": "exojax.rt.rtlayer",
    "rtransfer": "exojax.rt.rtransfer",
    "toon": "exojax.rt.toon",
    "twostream": "exojax.rt.twostream",
    "mie": "exojax.database.mie",
    "api": "exojax.database.api",
    "atomll": "exojax.database.atomll",
    "core_atom.io": "exojax.database.core_atom.io",
    "contdb": "exojax.database.contdb",
    "customapi": "exojax.database.customapi",
    "dbmanager": "exojax.database.dbmanager",
    "exomol": "exojax.database.exomol",
    "exomolhr": "exojax.database.exomolhr",
    "hitran": "exojax.database.hitran",
    "hitranapi": "exojax.database._common.hitranapi",
    "hitrancia": "exojax.database.hitrancia",
    "hminus": "exojax.database.hminus",
    "moldb": "exojax.database.moldb",
    "molinfo": "exojax.database.molinfo",
    "multimol": "exojax.database.multimol",
    "nonair": "exojax.database.nonair",
    "pardb": "exojax.database.pardb",
    "qstate": "exojax.database.molinfo",
    "limb_darkening": "exojax.postproc.limb_darkening",
    "response": "exojax.postproc.response",
    "specop": "exojax.postproc.specop",
    "spin_rotation": "exojax.postproc.spin_rotation",
    "OpaPremodit": "exojax.opacity.OpaPremodit",
    "OpaDirect": "exojax.opacity.OpaDirect",
    "OpaModit": "exojax.opacity.OpaModit",
}

# ────────────────────────────────────────────────────────────────────
# 2) Sub-module aliases  (e.g. import exojax.spec.opacalc as oc)
# ────────────────────────────────────────────────────────────────────

_SUBMODULES: Final[dict[str, str]] = {
    "opacalc": "exojax.opacity",
    "lpf": "exojax.opacity.lpf",
    "modit": "exojax.opacity.modit",  #
    "premodit": "exojax.opacity.premodit",  #
}


# ────────────────────────────────────────────────────────────────────
# 3) Lazy attribute resolution
# ────────────────────────────────────────────────────────────────────
def __getattr__(name: str):  # noqa: D401
    """Resolve deprecated names on first access (lazy import)."""
    target = _ALIASES.get(name)
    if target is None:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")

    mod_path, _, attr = target.rpartition(".")
    module = importlib.import_module(mod_path)
    obj = getattr(module, attr)

    # Cache for next time
    globals()[name] = obj
    return obj


# ────────────────────────────────────────────────────────────────────
# 4) Inject lightweight proxy sub-modules
# ────────────────────────────────────────────────────────────────────
class _Moved(types.ModuleType):
    """Proxy module that forwards everything to *new_fqn*."""

    def __init__(self, new_fqn: str, old_fqn: str):
        super().__init__(old_fqn)
        self._new_fqn = new_fqn
        self.__doc__ = f"Deprecated alias for `{new_fqn}`."

    def __getattr__(self, item):
        return getattr(importlib.import_module(self._new_fqn), item)

    def __dir__(self):
        return dir(importlib.import_module(self._new_fqn))


for _old, _new in _SUBMODULES.items():
    full_old = f"{__name__}.{_old}"
    if full_old not in sys.modules:
        shim = _Moved(_new, full_old)
        sys.modules[full_old] = shim
        globals()[_old] = shim

# ────────────────────────────────────────────────────────────────────
# 5) Public names for tab completion / dir()
# ────────────────────────────────────────────────────────────────────
__all__ = sorted(set(_ALIASES) | set(_SUBMODULES))
