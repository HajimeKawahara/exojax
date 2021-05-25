"""Astronomical messy functions

   * Astronomy is the oldest science in the world.
   * That's why we need this module.

"""

import exojax.utils.constants as const
import numpy as np
import jax.numpy as jnp

def getjov_logg(Rp,Mp):
    """logg
    
    Args:
       Rp: radius in the unit of Jovian radius
       Mp: radius in the unit of Jovian mass

    Returns 
       logg

    """    

    #Mpcgs=Mp*const.MJ
    #Rpcgs=Rp*const.RJ    
    #return np.log10(const.G*Mpcgs/Rpcgs**2)
    return jnp.log10(2478.57730044555*Mp/Rp**2)
    
def getjov_gravity(Rp,Mp):
    """logg
    
    Args:
       Rp: radius in the unit of Jovian radius
       Mp: radius in the unit of Jovian mass

    Returns 
       gravity (cm/s2)

    """    
    #Mpcgs=Mp*const.MJ
    #Rpcgs=Rp*const.RJ    
    #return (const.G*Mpcgs/Rpcgs**2)
    return 2478.57730044555*Mp/Rp**2
