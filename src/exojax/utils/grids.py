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
    
    _check_even_number(N)
    grid_mode = check_scale_xsmode(xsmode) 
    grid = _set_grid(x0, x1, N, unit, grid_mode)
    nus = _set_nus(unit, grid)
    wav = nu2wav(nus, unit="AA")
    resolution = _set_resolution(grid_mode, nus)

    return nus, wav, resolution

def _set_grid(x0, x1, N, unit, grid_mode):
    if grid_mode == 'ESLOG':
        grid = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
    elif grid_mode == 'ESLIN':
        grid = _set_grid_eslin(unit,x0,x1,N)
    else:
        raise ValueError("unavailable xsmode/unit.")
    return grid

def _check_even_number(N):
    if np.mod(N,2)==1:
        msg = "Currently, only even number is allowed as N. "
        msg += "response.convolve_rigid_rotation requires this condition."       
        raise ValueError(msg)

def _set_nus(unit, grid):
    if unit == 'cm-1':
        nus = grid
    elif unit == 'nm' or unit == 'AA':
        nus = wav2nu(grid, unit)
    return nus

def _set_resolution(grid_mode, nus):
    if grid_mode == 'ESLOG':
        resolution = resolution_eslog(nus)
        minr = resolution
    elif grid_mode == 'ESLIN':
        minr, resolution, maxr = resolution_eslin(nus)
    warn_resolution(minr)
    return resolution

def _set_grid_eslin(unit,x0,x1,N):
    if unit == "cm-1":
        return np.linspace((x0), (x1), N, dtype=np.float64)
    else: 
        cx1, cx0 = wav2nu(np.array([x0, x1]), unit)
        return np.linspace((cx0), (cx1), N, dtype=np.float64)
    
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
