"""Utility Functions about Instruments.
"""

import numpy as np
from exojax.utils.constants import c
import warnings


def R2STD(resolution):
    warn_msg = "Use `resolution_to_gaussian_std` instead"
    warnings.warn(warn_msg, FutureWarning)
    return resolution_to_gaussian_std(resolution)


def resolution_to_gaussian_std(resolution):
    """compute Standard deveiation of Gaussian velocity distribution from
    spectral resolution.

    Args:
        resolution: spectral resolution R

    Returns:
        standard deviation of Gaussian velocity distribution (km/s)
    """
    return c / (2.0 * np.sqrt(2.0 * np.log(2.0)) * resolution)


def resolution_eslog(nu_grid):
    """spectral resolution for ESLOG.

    Args:
        nu_grid: wavenumber bin

    Returns:
        resolution
    """
    return (len(nu_grid) - 1) / np.log(nu_grid[-1] / nu_grid[0])


def resolution_eslin(nu_grid):
    """min max spectral resolution for ESLIN.

    Args:
        nu_grid: wavenumber bin

    Returns:
        min, approximate, max of the resolution
    """
    resolution = ((nu_grid[-1] + nu_grid[0]) / 2.0) / (
        (nu_grid[-1] - nu_grid[0]) / len(nu_grid)
    )
    return (
        nu_grid[0] / (nu_grid[1] - nu_grid[0]),
        resolution,
        nu_grid[-1] / (nu_grid[-1] - nu_grid[-2]),
    )


def nx_even_from_resolution_eslog(nu0, nu1, resolution):
    """Compute the even number of wavenumber grid for a given resolution for ESLOG

    Args:
        nu0 (float): wavenumber min
        nu1 (float): wavenumber max
        resolution (float): resolution

    Returns:
        int: the even number of wavenumber grid for a given resolution
    """
    N = np.ceil(resolution * np.log(nu1 / nu0))
    return int(np.ceil(N / 2.0) * 2  )
