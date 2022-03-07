"""Saturation Vapor Pressure."""
import jax.numpy as jnp


def Psat_enstatite_AM01(T):
    """Saturation Vapor Pressure for Enstatite (MgSiO3)

    Note:
       Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
       T: temperature (K)

    Returns:
       saturation vapor pressure (bar)
    """
    return jnp.exp(25.37-58663./T)


def Psat_Fe_solid(T):
    """Saturation Vapor Pressure for Solid Fe (Fe)

    Note:
       Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
       T: temperature (K)

    Returns:
       saturation vapor pressure (bar)
    """
    return jnp.exp(15.71-47664./T)


def Psat_Fe_liquid(T):
    """Saturation Vapor Pressure for liquid Fe (Fe)

    Note:
       Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
       T: temperature (K)

    Returns:
       saturation vapor pressure (bar)
    """
    return jnp.exp(9.86-37120./T)
