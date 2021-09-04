"""Utility Functions about Instruments


"""

import numpy as np
from exojax.utils.constants import c

def R2STD(R):
    """ compute Standard deveiation of Gaussian from spectral resolution

    Args:
       R: spectral resolution R

    Returns:
       STD: Gaussian STD


    """
    return c/(2.0*np.sqrt(2.0*np.log(2.0))*R)
