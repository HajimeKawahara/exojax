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


def resolution_eslog(nu_grid, *, definition="log"):
    """Compute the spectral resolution of an ESLOG grid.

    Args:
        nu_grid: wavenumber bin
        definition: Resolution definition. ``"log"`` returns the inverse
            logarithmic spacing. ``"pointwise"`` returns the minimum of
            ``nu_grid[:-1] / np.diff(nu_grid)``.

    Returns:
        resolution
    """
    if definition == "log":
        return (len(nu_grid) - 1) / np.log(nu_grid[-1] / nu_grid[0])
    if definition == "pointwise":
        return np.min(nu_grid[:-1] / np.diff(nu_grid))
    raise ValueError("definition must be 'log' or 'pointwise'.")


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


def nx_even_from_resolution_eslog(nu0, nu1, resolution, *, definition="log"):
    """Compute an even ESLOG grid size for a given resolution.

    Args:
        nu0 (float): wavenumber min
        nu1 (float): wavenumber max
        resolution (float): resolution
        definition: Resolution definition. ``"log"`` preserves the existing
            inverse-log-spacing behavior. ``"pointwise"`` returns the
            smallest even grid size whose adjacent-point resolving power is
            at least ``resolution``.

    Returns:
        int: the even number of wavenumber grid for a given resolution
    """
    if definition == "log":
        number_of_points = np.ceil(resolution * np.log(nu1 / nu0))
    elif definition == "pointwise":
        log_span = np.log(nu1) - np.log(nu0)
        number_of_intervals = np.ceil(
            log_span / np.log1p(1.0 / resolution)
        )
        number_of_points = number_of_intervals + 1
    else:
        raise ValueError("definition must be 'log' or 'pointwise'.")
    return int(np.ceil(number_of_points / 2.0) * 2)
