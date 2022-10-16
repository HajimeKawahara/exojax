"""
    * setrt will be removed
"""
import warnings
from exojax.utils.grids import wavenumber_grid

def gen_wavenumbere_grids(x0, x1, N, unit, xsmode):
    warn_msg = " Use `utils.grids.wavenumber_grid` instead"
    warnings.warn(warn_msg, DeprecationWarning)
    return wavenumber_grid(x0, x1, N, unit, xsmode)

