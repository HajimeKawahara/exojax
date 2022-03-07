"""Atmospheric profile function."""

from exojax.utils.constants import kB, m_u
import jax.numpy as jnp


def Hatm(g, T, mu):
    """pressure scale height assuming an isothermal atmosphere.

    Args:
        g: gravity acceleration (cm/s2)
        T: isothermal temperature (K)
        mu: mean molecular weight

    Returns:
        pressure scale height (cm)
    """

    return kB*T/(m_u*mu*g)


def atmprof_gray(Parr, g, kappa, Tint):
    """

    Args:
        Parr: pressure array (bar)
        g: gravity (cm/s2)
        kappa: infrared opacity 
        Tint: temperature equivalence of the intrinsic energy flow

    """

    tau = Parr*1.e6*kappa/g
    Tarr = (0.75*Tint**4*(2.0/3.0+tau))**0.25
    return Tarr


def atmprof_Guillot(Parr, g, kappa, gamma, Tint, Tirr, f=0.25):
    """

    Notes:
        Guillot (2010) Equation (29)

    Args:
        Parr: pressure array (bar)
        g: gravity (cm/s2)
        kappa: thermal/IR opacity (kappa_th in Guillot 2010)
        gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
        Tint: temperature equivalence of the intrinsic energy flow
        Tirr: temperature equivalence of the irradiation
        f = 1 at the substellar point, f = 1/2 for a day-side average 
            and f = 1/4 for an averaging over the whole planetary surface

    Returns:
        Tarr

    """
    tau = Parr*1.e6*kappa/g  # Equation (51)
    invsq3 = 1.0/jnp.sqrt(3.0)
    fac = 2.0/3.0 + invsq3*(1.0/gamma + (gamma - 1.0/gamma)
                            * jnp.exp(-gamma*tau/invsq3))
    Tarr = (0.75*Tint**4*(2.0/3.0+tau) + 0.75*Tirr**4*f*fac)**0.25

    return Tarr


def Teq2Tirr(Teq, Tint):
    """Tirr from equilibrium temperature and intrinsic temperature.

    Args:
       Teq: equilibrium temperature
       Tint: intrinsic temperature

    Return:
       Tirr: iradiation temperature

    Note:
       Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (2.0**0.5)*Teff


def Teff2Tirr(Teff, Tint):
    """Tirr from effective temperature and intrinsic temperature.

    Args:
       Teff: effective temperature
       Tint: intrinsic temperature

    Return:
       Tirr: iradiation temperature

    Note:
       Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (4.0*Teff**4 - Tint**4)**0.25
