"""generate various grids
"""
from exojax.spec.unitconvert import nu2wav, wav2nu
from exojax.spec.check_nugrid import check_scale_xsmode, warn_resolution
from exojax.utils.instfunc import resolution_eslog, resolution_eslin
from exojax.utils.constants import c
import jax.numpy as jnp
import numpy as np
import warnings

def wavenumber_grid(x0, x1, N, unit='cm-1', xsmode='lpf'):
    """generating the recommended wavenumber grid based on the cross section
    computation mode.

    Args:
        x0: start wavenumber (cm-1) or wavelength (nm) or (AA)
        x1: end wavenumber (cm-1) or wavelength (nm) or (AA)
        N: the number of the wavenumber grid (even number)
        unit: unit of the input grid
        xsmode: cross section computation mode (lpf, dit, modit, premodit)

    Returns:
        wavenumber grid evenly spaced in log space
        corresponding wavelength grid (AA)
        resolution
    """
    
    if np.mod(N,2)==1:
        msg = "Currently, only even number is allowed as N. "
        msg += "response.convolve_rigid_rotation requires this condition."       
        raise ValueError(msg)
        
    if check_scale_xsmode(xsmode) == 'ESLOG':
        grid = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
    elif check_scale_xsmode(xsmode) == 'ESLIN' and unit == 'cm-1':
        warnings.warn("ESLIN is not recommended. Consider to use ESLOG instead.", UserWarning)
        grid = np.linspace((x0), (x1), N, dtype=np.float64)
    else:
        raise ValueError("unavailable xsmode/unit.")
        
    if unit == 'cm-1':
        nus = grid
    elif unit == 'nm' or unit == 'AA':
        nus = wav2nu(grid, unit)
        
    wav = nu2wav(nus, unit="AA")
        
    if check_scale_xsmode(xsmode) == 'ESLOG':
        resolution = resolution_eslog(nus)
        minr = resolution
    elif check_scale_xsmode(xsmode) == 'ESLIN':
        minr, resolution, maxr = resolution_eslin(nus)
        
    warn_resolution(minr)

    return nus, wav, resolution



def velocity_grid(resolution, vmax):
    """generate velocity grid for a rigid rotation

    Args:
        resolution: spectral resolution
        vmax: maximum velocity (or Vsini) allowed (km/s)

    Returns:
        1D array: delta velocity grid 
    """
    dv = delta_velocity_from_resolution(resolution)
    Nk = (vmax / dv) + 1
    Nk = Nk.astype(int)
    return dv * np.arange(-Nk, Nk + 1)


def delta_velocity_from_resolution(resolution):
    """delta velocity from spectral resolution R

    Args:
        resolution : spectral resolution

    Note: 
        See also [#294](https://github.com/HajimeKawahara/exojax/issues/294) and exojax/tests/figures/functions/delta_velocity_comp.py

    Returns:
        delta velocity
    """
    return c * jnp.log1p(1.0 / resolution)
