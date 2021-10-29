"""Utility Functions about Instruments


"""

import numpy as np
from exojax.utils.constants import c

def R2STD(R):
    """ compute Standard deveiation of Gaussian velocity distribution from spectral resolution

    Args:
       R: spectral resolution R

    Returns:
       beta (km/s) standard deviation of Gaussian velocity distribution


    """
    return c/(2.0*np.sqrt(2.0*np.log(2.0))*R)

def resolution_eslog(nu):
    """spectral resolution for ESLOG

    Args:
       nu: wavenumber bin

    Returns:
       resolution

    """
    return (len(nu)-1)/np.log(nu[-1]/nu[0])
