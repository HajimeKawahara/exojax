__all__ = []

__version__ = '1.6'
__uri__ = ''
__author__ = 'ExoJAX contributors'
__email__ = 'divrot@gmail.com'
__license__ = ''
__description__ = 'auto-differentiable spectral modules in exojax'

from exojax.spec.hitran import (line_strength, doppler_sigma,
                                gamma_natural, normalized_doppler_sigma)

from exojax.opacity.lpf import (
    hjert,
    voigt,
    voigtone,
    vvoigt,
)

from exojax.opacity.make_numatrix import (
    make_numatrix0, )

"""
Compatibility layer for legacy imports.

`exojax.spec` will be **removed in v1.0**.  
Please import the new packages instead:

    exojax.opacity.*  (for opacity calculators)
    exojax.rt.*       (for radiative-transfer helpers)

This file keeps old code running during the deprecation window.
"""
from importlib import import_module
import sys
import warnings

# --- emit warning once per Python session ---
warnings.warn(
    "`exojax.spec` is deprecated and will be removed in v3.0. "
    "Update your imports to `exojax.opacity` or `exojax.rt`.",
    DeprecationWarning,
    stacklevel=2,
)

# --- helper to register an alias ---
def _alias(old_subname: str, new_fqdn: str) -> None:
    """
    Map `exojax.spec.<old_subname>` → NEW fully-qualified module.

    Parameters
    ----------
    old_subname : str
        e.g. "modit"
    new_fqdn : str
        e.g. "exojax.opacity.modit"
    """
    new_mod = import_module(new_fqdn)
    old_key = f"{__name__}.{old_subname}"
    sys.modules[old_key] = new_mod
    globals()[old_subname] = new_mod


# ---- alias table (extend as needed) ----
_ALIAS_MAP = {
    # opacity layer
    "opacalc":        "exojax.opacity.opacalc",
    "initspec":       "exojax.opacity.initspec",
    "premodit":       "exojax.opacity.premodit",
    "modit":          "exojax.opacity.modit",
    "lpf":            "exojax.opacity.lpf",
    "set_ditgrid":    "exojax.opacity.set_ditgrid",
    "optgrid":        "exojax.opacity.optgrid",
    "lbd":            "exojax.opacity.lbd",
    "lsd":            "exojax.opacity.lsd",
    "lbderror":       "exojax.opacity.lbderror",
    "dit":            "exojax.opacity.dit",
    "ditkernel":      "exojax.opacity.ditkernel",
    "make_numatrix":  "exojax.opacity.make_numatrix",
    "opacont":        "exojax.opacity.opacont",
    "mie":            "exojax.opacity.mie",
    "rayleigh":       "exojax.opacity.rayleigh",
    "generate_elower_grid_trange": "exojax.opacity.generate_elower_grid_trange",
    "profconv":       "exojax.opacity.profconv",
    "lpffilter":      "exojax.opacity.lpffilter",
    "chord":          "exojax.rt.chord",
    "atmrt":          "exojax.rt.atmrt",
    "opart":          "exojax.rt.opart",
    "layeropacity":   "exojax.rt.layeropacity",
    "planck":         "exojax.rt.planck",
    "rtlayer":        "exojax.rt.rtlayer",
    "rtransfer":      "exojax.rt.rtransfer",
    "toon":           "exojax.rt.toon",
    "twostream":      "exojax.rt.twostream",
}

for _old, _new in _ALIAS_MAP.items():
    _alias(_old, _new)
