"""Utility Functions about Instruments.
"""

import numpy as np
from exojax.utils.constants import c
import warnings


def R2STD(resolution):
    warn_msg = "Use `resolution_to_gaussian_std` instead"
    warnings.warn(warn_msg, DeprecationWarning)
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


def resolution_eslog(nu):
    """spectral resolution for ESLOG.

    Args:
       nu: wavenumber bin

    Returns:
       resolution
    """
    return (len(nu) - 1) / np.log(nu[-1] / nu[0])


def resolution_eslin(nu):
    """min max spectral resolution for ESLIN.

    Args:
       nu: wavenumber bin

    Returns:
       min, approximate, max of the resolution
    """
    resolution = ((nu[-1] + nu[0]) / 2.0) / ((nu[-1] - nu[0]) / len(nu))
    return nu[0] / (nu[1] - nu[0]), resolution, nu[-1] / (nu[-1] - nu[-2])
