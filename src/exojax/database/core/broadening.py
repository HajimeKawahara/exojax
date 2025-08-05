import numpy as np
import jax.numpy as jnp
from jax import jit
from exojax.utils.constants import Patm
from exojax.utils.constants import Tref_original
from exojax.utils.constants import ccgs
from exojax.utils.constants import hcperk


def line_strength_from_Einstein_coeff(A, g, nu_lines, elower, QTref):
    """Reference Line Strength in Tref=296K, S0 from Einstein coefficient.

    Note:
        This function is not used in other codes in ExoJAX.
        But it can be used for the conversion of the line strength from the original ExoMol form
        into HITRAN form.

    Args:
        A: Einstein coefficient (s-1)
        g: the upper state statistical weight
        nu_lines: line center wavenumber (cm-1)
        elower: elower
        QTref: partition function Q(Tref)
        Mmol: molecular mass (normalized by m_u)

    Returns:
        Line strength (cm)
    """
    line_strength_ref = (
        -A
        * g
        * np.exp(-hcperk * elower / Tref_original)
        * np.expm1(-hcperk * nu_lines / Tref_original)
        / (8.0 * np.pi * ccgs * nu_lines**2 * QTref)
    )
    return line_strength_ref


def gamma_exomol(P, T, n_air, alpha_ref):
    """gamma factor by a pressure broadening.

    Args:
        P: pressure (bar)
        T: temperature (K)
        n_air: coefficient of the  temperature  dependence  of  the  air-broadened halfwidth
        alpha_ref: broadening parameter

    Returns:
        gamma: pressure gamma factor (cm-1)
    """
    gamma = alpha_ref * P * (Tref_original / T) ** n_air
    return gamma


@jit
def gamma_hitran(P, T, Pself, n_air, gamma_air_ref, gamma_self_ref):
    """gamma factor by a pressure broadening.

    Args:
        P: pressure (bar)
        T: temperature (K)
        Pself: partial pressure (bar)
        n_air: coefficient of the temperature dependence of the air-broadened halfwidth
        gamma_air_ref: gamma air
        gamma_self_ref: gamma self

    Returns:
        gamma: pressure gamma factor (cm-1)
    """
    Tref = Tref_original  # reference tempearture (K)
    gamma = (Tref / T) ** n_air * (
        gamma_air_ref * ((P - Pself) / Patm) + gamma_self_ref * (Pself / Patm)
    )
    return gamma


@jit
def gamma_natural(A):
    """gamma factor by natural broadning.

    1/(4 pi c) = 2.6544188e-12 (cm-1 s)

    Args:
        A: Einstein A-factor (1/s)

    Returns:
        gamma_natural: natural width (cm-1)
    """
    return 2.6544188e-12 * A


@jit
def doppler_sigma(nu_lines, T, M):
    """Dopper width (sigmaD)

    Note:
        c3 is sqrt(kB/m_u)/c

    Args:
        nu_lines: line center wavenumber (cm-1)
        T: temperature (K)
        M: atom/molecular mass

    Returns:
        sigma: doppler width (standard deviation) (cm-1)
    """
    c3 = 3.0415595e-07
    return c3 * jnp.sqrt(T / M) * nu_lines


@jit
def normalized_doppler_sigma(T, M, R):
    """Normalized Dopper width (nsigmaD) by wavenumber difference at line
    centers.

    Note:
        This quantity is used in MODIT. c3 is sqrt(kB/m_u)/c

    Args:
        T: temperature (K)
        M: atom/molecular mass
        R: spectral resolution

    Returns:
        nsigma: normalized Doppler width (standard deviation)
    """
    c3 = 3.0415595e-07
    return c3 * jnp.sqrt(T / M) * R
