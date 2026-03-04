"""Astronomical messy functions.

* Astronomy is the oldest science in the world.
* That's why we need this module.
"""

from exojax.utils.constants import gJ, gE
from exojax.utils.constants import loggJ

import jax.numpy as jnp

def square_radius_from_mass_logg(Mp, logg):
    """square of radius from mass and logg.

    Args:
        Mp: mass in the unit of Jupiter mass
        logg: logg (log10 gravity in cgs)

    Returns:
        Rp**2 in the unit of sqaured Jupiter radius

    """
    return Mp * 10**(loggJ - logg)


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


def gravity_earth(Rp, Mp):
    """gravity in cgs from radius and mass in the Terrestrial unit.

    Args:
        Rp: radius in the unit of Earth radius
        Mp: radius in the unit of Earth mass

    Returns:
        gravity (cm/s2)

    Note:
        Mpcgs=Mp*const.ME, Rpcgs=Rp*const.RE
        then gravity is given by (const.G*Mpcgs/Rpcgs**2)
    """
    return gE * Mp / Rp**2
