"""generate various grids
"""
from exojax.spec.unitconvert import nu2wav, wav2nu
from exojax.utils.instfunc import resolution_eslog, resolution_eslin
from exojax.utils.constants import c
import jax.numpy as jnp
import numpy as np
import warnings


def wavenumber_grid(x0, x1, N, xsmode, wavelength_order="descending", unit='cm-1'):
    """generating the recommended wavenumber grid based on the cross section
    computation mode.

    Args:
        x0: start wavenumber (cm-1) or wavelength (nm) or (AA)
        x1: end wavenumber (cm-1) or wavelength (nm) or (AA)
        N: the number of the wavenumber grid (even number)
        xsmode: cross section computation mode (lpf, dit, modit, premodit)
        wavlength order: wavelength order: "ascending" or "descending"
        unit: unit of the input grid
        
    Note:
        The wavenumber (nus) and wavelength (wav) grids are in ascending orders. 
        Therefore, wav[-1] corresponds to the wavelength of nus[0].
        ESLIN sets evenly-spaced linear grid in wavenumber space while ESLOG sets 
        evenly-spaced log grid both in wavenumber and wavelength spaces. 

    Returns:
        nu_grid: wavenumber grid evenly spaced in log space in ascending order (nus)
        wav: corresponding wavelength grid (AA) in ascending order (wav). wav[-1] corresponds to nus[0]
        resolution: spectral resolution
    """
    print("xsmode = ", xsmode)
    _check_even_number(N)
    grid_mode = check_scale_xsmode(xsmode)
    grid, unit = _set_grid(x0, x1, N, unit, grid_mode)
    nu_grid = _set_nus(unit, grid)

    _warning_wavelength_order(wavelength_order)

    wav = nu2wav(nu_grid, wavelength_order=wavelength_order, unit="AA")
    resolution = grid_resolution(grid_mode, nu_grid)
    return nu_grid, wav, resolution

def _warning_wavelength_order(wavelength_order):
    """this is temporary special warning on wavelenght order

    Args:
        wavlength order: wavelength order: "ascending" or "descending"
    """
    print("======================================================================")
    print("We changed the policy of the order of wavenumber/wavelength grids")
    print("wavenumber grid should be in ascending order and now ")
    print("users can specify the order of the wavelength grid by themselves.")
    print("Your wavelength grid is in *** ", wavelength_order, " *** order")
    print("This might causes the bug if you update ExoJAX. ")
    print("Note that the older ExoJAX assumes ascending order as wavelength grid.")
    print("======================================================================")


def _set_grid(x0, x1, N, unit, grid_mode):
    if grid_mode == 'ESLOG':
        grid = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
    elif grid_mode == 'ESLIN':
        grid, unit = _set_grid_eslin(unit, x0, x1, N)
    else:
        raise ValueError("unavailable xsmode/unit.")
    return grid, unit


def _check_even_number(N):
    if np.mod(N, 2) == 1:
        msg = "Currently, only even number is allowed as N. "
        msg += "response.convolve_rigid_rotation requires this condition."
        raise ValueError(msg)


def _set_nus(unit, grid):
    if unit == 'cm-1':
        nus = grid
    elif unit == 'nm' or unit == 'AA':
        nus = wav2nu(grid, unit)
    return nus


def grid_resolution(grid_mode, nus):
    if grid_mode == 'ESLOG':
        resolution = resolution_eslog(nus)
        minr = resolution
    elif grid_mode == 'ESLIN':
        minr, resolution, _ = resolution_eslin(nus)
    warn_resolution(minr)
    return resolution


def _set_grid_eslin(unit, x0, x1, N):
    if unit == "cm-1":
        return np.linspace((x0), (x1), N, dtype=np.float64), unit
    else:
        cx0, cx1 = wav2nu(np.array([x0, x1]), unit)
        unit = 'cm-1'
        return np.linspace((cx0), (cx1), N, dtype=np.float64), unit


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


def warn_resolution(resolution, crit=700000.0):
    """warning poor resolution.

    Args:
        resolution: spectral resolution
        crit: critical resolution
    """
    if resolution < crit:
        warnings.warn('Resolution may be too small. R=' + str(resolution),
                      UserWarning)


def check_scale_xsmode(xsmode):
    """checking if the scale of xsmode assumes ESLOG(log) or ESLIN(linear)

    Args:
       xsmode: xsmode

    Return:
       ESLOG/ESLIN/UNKNOWN
    """
    def _add_upper_case(strlist):
        return strlist + [x.upper() for x in strlist]

    eslog_list = _add_upper_case(['lpf', 'modit', 'premodit', 'presolar'])
    eslin_list = _add_upper_case(['dit'])
    if xsmode in eslog_list:
        print('xsmode assumes ESLOG in wavenumber space: mode=' + str(xsmode))
        return 'ESLOG'
    elif xsmode in eslin_list:
        print('xsmode assumes ESLIN in wavenumber space: mode=' + str(xsmode))
        return 'ESLIN'
    else:
        warnings.warn("unknown xsmode.", UserWarning)
        return 'UNKNOWN'


def check_eslog_wavenumber_grid(nus,
                                crit1=1.e-5,
                                crit2=1.e-14,
                                gridmode='ESLOG'):
    """checking if wavenumber_grid is evenly spaced in a logarithm scale (ESLOG) or a
    liner scale (ESLIN)

    Args:
       nus: wavenumber grid
       crit1: criterion for the maximum deviation of log10(nu)/median(log10(nu)) from ESLOG
       crit2: criterion for the maximum deviation of log10(nu) from ESLOG
       gridmode: ESLOG or ESLIN

    Returns:
       True (wavenumber grid is ESLOG) or False (not)
    """

    q = np.log10(nus)
    p = q[1:] - q[:-1]
    w = (p - np.mean(p))
    val1 = np.max(np.abs(w)) / np.median(p)
    val2 = np.max(np.abs(w))

    return (val1 < crit1 and val2 < crit2)


