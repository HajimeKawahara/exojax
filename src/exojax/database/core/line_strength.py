import jax.numpy as jnp
import numpy as np
from jax import jit

from exojax.utils.constants import Tref_original
from exojax.utils.constants import ccgs
from exojax.utils.constants import hcperk


@jit
def line_strength(T, logsij0, nu_lines, elower, qr, Tref):
    """Line strength as a function of temperature, JAX/XLA compatible

    Notes:
        Tref=296.0 (default) in moldb, but it might have been changed by OpaPremodit.

    Args:
        T: temperature (K)
        logsij0: log(Sij(Tref)) (Tref=296K)
        nu_lines: line center wavenumber (cm-1)
        elower: elower
        qr: partition function ratio qr(T) = Q(T)/Q(Tref)
        Tref: reference temperature

    Returns:
        Sij(T): Line strength (cm)
    """
    expow = logsij0 - hcperk * (elower / T - elower / Tref)
    fac = (1.0 - jnp.exp(-hcperk * nu_lines / T)) / (
        1.0 - jnp.exp(-hcperk * nu_lines / Tref)
    )
    # expow=logsij0-hcperk*elower*(1.0/T-1.0/Tref)
    # fac=jnp.expm1(-hcperk*nu_lines/T)/jnp.expm1(-hcperk*nu_lines/Tref)
    return jnp.exp(expow) / qr * fac


def line_strength_numpy(T, Sij0, nu_lines, elower, qr, Tref=Tref_original):
    """Line strength as a function of temperature, numpy version

    Args:
        T: temperature (K)
        Sij0: line strength at Tref=296K
        elower: elower
        nu_lines: line center wavenumber
        qr : partition function ratio qr(T) = Q(T)/Q(Tref)
        Tref: reference temeparture

    Returns:
        line strength at Ttyp
    """
    return (
        Sij0
        * np.exp(-hcperk * elower * (1.0 / T - 1.0 / Tref))
        * np.expm1(-hcperk * nu_lines / T)
        / np.expm1(-hcperk * nu_lines / Tref)
        / qr  # Apply qr (jnp array) last to minimize rounding errors in 32bit mode.
    )


def Einstein_coeff_from_line_strength(nu_lines, Sij, elower, g, Q, T):
    """Einstein coefficient from line strength

    Args:
        nu_lines (float): line center wavenumber (cm-1)
        Sij (float): line strength (cm)
        elower (float): elower
        g (float): upper state statistical weight
        Q (float): partition function
        T (float): temperature

    Returns:
        Einstein coefficient (s-1)
    """    
    Aij = - ((Sij * 8.0 * np.pi * ccgs * nu_lines**2 * Q) 
            / g
            / np.exp(-hcperk * elower / T) 
            / np.expm1(-hcperk * nu_lines / T))
    return Aij
