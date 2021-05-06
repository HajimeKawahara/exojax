"""Astronomical messy functions

   * Astronomy is the oldest science in the world.
   * That's why we need this module.

"""

import exojax.utils.constants as const
import numpy as np

def getjov_logg(Rp,Mp):
    """logg
    
    Args:
       Rp: radius in the unit of Jovian radius
       Mp: radius in the unit of Jovian mass

    Returns 
       logg

    """    
    Mpcgs=Mp*const.MJ
    Rpcgs=Rp*const.RJ    
    return np.log10(const.G*Mpcgs/Rpcgs**2)
