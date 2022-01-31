"""Functions about ideal gas."""

from exojax.utils.constants import kB


def number_density(Parr, Tarr):
    """number density of ideal gas in cgs.

    Args:
       Parr: pressure array (bar)
       Tarr: temperature array (K)

    Returns:
       number density (1/cm3)
    """

    return (Parr*1.e6)/(kB*Tarr)
