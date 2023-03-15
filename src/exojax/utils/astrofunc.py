"""Astronomical messy functions.

* Astronomy is the oldest science in the world.
* That's why we need this module.
"""

from exojax.utils.constants import gJ
import jax.numpy as jnp


def logg_jupiter(Rp, Mp):
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
    return jnp.log10(gravity_jupiter(Rp, Mp))


def gravity_jupiter(Rp, Mp):
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
    return gJ * Mp / Rp**2
