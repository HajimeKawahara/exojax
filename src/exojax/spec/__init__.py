__all__ = []

__version__ = '2.0'
__uri__ = ''
__author__ = 'ExoJAX contributors'
__email__ = 'divrot@gmail.com'
__license__ = ''
__description__ = 'auto-differentiable spectral modules in exojax'


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
    "mie":            "exojax.database.mie",
    "api":            "exojax.database.api",
    "atomll":        "exojax.database.atomll",
    "atomllapi":     "exojax.database.atomllapi",
    "contdb":        "exojax.database.contdb",
    "customapi":     "exojax.database.customapi",
    "dbmanager":     "exojax.database.dbmanager", 
    "exomol":        "exojax.database.exomol",
    "exomolhr":      "exojax.database.exomolhr",
    "hitran":        "exojax.database.hitran",
    "hitranapi":     "exojax.database.hitranapi",
    "hitrancia":     "exojax.database.hitrancia",
    "hminus":        "exojax.database.hminus",
    "moldb":         "exojax.database.moldb",
    "molinfo":       "exojax.database.molinfo",
    "multimol":      "exojax.database.multimol",
    "nonair":        "exojax.database.nonair",
    "pardb":         "exojax.database.pardb",
    "qstate":        "exojax.database.qstate",
    "limb_darkening":"exojax.postproc.limb_darkening",
    "response":      "exojax.postproc.response",
    "specop":        "exojax.postproc.specop",
    "spin_rotation": "exojax.postproc.spin_rotation",
}


for _old, _new in _ALIAS_MAP.items():
    _alias(_old, _new)

__all__ = list(_ALIAS_MAP)