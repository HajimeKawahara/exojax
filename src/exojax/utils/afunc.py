"""Astronomical messy functions.

* Astronomy is the oldest science in the world.
* That's why we need this module.
"""

import exojax.utils.constants as const
import numpy as np
import jax.numpy as jnp


def getjov_logg(Rp, Mp):
    """logg from radius and mass in the Jovian unit.

    Args:
       Rp: radius in the unit of Jovian radius
       Mp: radius in the unit of Jovian mass

    Returns:
       logg

    Note:
       Mpcgs=Mp*const.MJ, Rpcgs=Rp*const.RJ,
       then logg is given by log10(const.G*Mpcgs/Rpcgs**2)
    """
    return jnp.log10(2478.57730044555*Mp/Rp**2)


def getjov_gravity(Rp, Mp):
    """gravity in cgs from radius and mass in the Jovian unit.

    Args:
       Rp: radius in the unit of Jovian radius
       Mp: radius in the unit of Jovian mass

    Returns:
       gravity (cm/s2)

    Note:
       Mpcgs=Mp*const.MJ, Rpcgs=Rp*const.RJ
       then gravity is given by (const.G*Mpcgs/Rpcgs**2)
    """
    return 2478.57730044555*Mp/Rp**2
