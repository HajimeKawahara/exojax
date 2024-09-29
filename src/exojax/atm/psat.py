"""Saturation Vapor Pressure."""

import jax.numpy as jnp
from exojax.utils.constants import Tc_water, Ttp_water


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
    Tcelcius = T - Tc_water

    return 6.1094e-3 * jnp.exp(17.625 * (Tcelcius) / ((Tcelcius) + 243.04))


def psat_water_AM01(T):
    """
    Saturation Vapor Pressure for water (but for T<1048K)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata and updated
        Buck 1981, 1996

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    Tcelcius = T - Tc_water
    Tcrit = 1048  # K
    return jnp.where(
        T > Tcrit,
        600.0,
        jnp.where(
            T < Tc_water,
            _psat_water_buck_ice(Tcelcius),
            _psat_water_buck_liquid(Tcelcius),
        ),
    )


def _psat_water_buck_ice(T):
    return 0.0061115 * jnp.exp((23.036 * T - T**2 / 333.7) / (T + 279.82))


def _psat_water_buck_liquid(T):
    # return 0.0061121*jnp.exp((18.729*T - T**2/227.3)/(T + 257.87)) AM Appendix but old
    return 0.0061121 * jnp.exp(
        (18.678 * T - T**2 / 234.5) / (T + 257.14)
    )  # from wikipedia (updated)


def psat_ammonia_AM01(T):
    """Saturation Vapor Pressure for Ammonia

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(10.53 - 2161.0 / T - 86596.0 / T**2)


def psat_enstatite_AM01(T):
    """Saturation Vapor Pressure for Enstatite (MgSiO3)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata, originally from  Barshay and Lewis (1976)

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(25.37 - 58663.0 / T)


def psat_Fe_AM01(T):
    """
    Saturation Vapor Pressure for iron

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A3), originally from  Barshay and Lewis (1976)


    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    Tc_Fe = 1800.0

    return jnp.where(
        T > Tc_Fe,
        _psat_Fe_liquid(T),
        _psat_Fe_solid(T),
    )
    # return jnp.exp(25.37 - 58663.0 / T)


def _psat_Fe_solid(T):
    """Saturation Vapor Pressure for Solid Fe (Fe)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(15.71 - 47664.0 / T)


def _psat_Fe_liquid(T):
    """Saturation Vapor Pressure for liquid Fe (Fe)

    Note:
        Taken from Ackerman and Marley 2001 Appendix A (A4) see also their errata.

    Args:
        T: temperature (K)

    Returns:
        saturation vapor pressure (bar)
    """
    return jnp.exp(9.86 - 37120.0 / T)
