"""Saturation Vapor Pressure."""
import jax.numpy as jnp


def psat_water_Magnus(T):
    """
    Saturation Vapor Pressure for water (Magnus, or August-Roche-Magnus)

    Note:
        See https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#August%E2%80%93Roche%E2%80%93Magnus_approximation

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return 6.1094e-3*jnp.exp(17.625*T/(T + 243.04))


def psat_enstatite_AM01(T):
    """Saturation Vapor Pressure for Enstatite (MgSiO3)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(25.37 - 58663.0 / T)


def psat_Fe_solid(T):
    """Saturation Vapor Pressure for Solid Fe (Fe)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(15.71 - 47664.0 / T)


def psat_Fe_liquid(T):
    """Saturation Vapor Pressure for liquid Fe (Fe)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(9.86 - 37120.0 / T)
