"""Utility Functions about Instruments."""

import numpy as np
from exojax.utils.constants import c


def R2STD(R):
    """compute Standard deveiation of Gaussian velocity distribution from
    spectral resolution.

    Args:
       R: spectral resolution R

    Returns:
       beta (km/s) standard deviation of Gaussian velocity distribution
    """
    return c/(2.0*np.sqrt(2.0*np.log(2.0))*R)


def resolution_eslog(nu):
    """spectral resolution for ESLOG.

    Args:
       nu: wavenumber bin

    Returns:
       resolution
    """
    return (len(nu)-1)/np.log(nu[-1]/nu[0])


def resolution_eslin(nu):
    """min max spectral resolution for ESLIN.

    Args:
       nu: wavenumber bin

    Returns:
       min, approximate, max of the resolution
    """
    return nu[0]/(nu[1]-nu[0]), ((nu[-1]+nu[0])/2.0)/((nu[-1]-nu[0])/len(nu)), nu[-1]/(nu[-1]-nu[-2])


if __name__ == '__main__':
    nus = np.linspace(1000, 2000, 1000)
    print(resolution_eslin(nus))
