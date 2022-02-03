from jax import jit
import jax.numpy as jnp
from exojax.utils.constants import hcperk


@jit
def SijT(T, logsij0, nu_lines, elower, qT):
    """Line strength as a function of temperature.

    Args:
       T: temperature (K)
       logsij0: log(Sij(Tref)) (Tref=296K)
       nu_lines: line center wavenumber (cm-1)
       elower: elower
       qT: Q(T)/Q(Tref)

    Returns:
       Sij(T): Line strength (cm)
    """
    Tref = 296.0  # reference tempearture (K)
    expow = logsij0-hcperk*(elower/T-elower/Tref)
    fac = (1.0-jnp.exp(-hcperk*nu_lines/T)) / \
        (1.0-jnp.exp(-hcperk*nu_lines/Tref))
    # expow=logsij0-hcperk*elower*(1.0/T-1.0/Tref)
    # fac=jnp.expm1(-hcperk*nu_lines/T)/jnp.expm1(-hcperk*nu_lines/Tref)
    return jnp.exp(expow)/qT*fac


@jit
def gamma_hitran(P, T, Pself, n_air, gamma_air_ref, gamma_self_ref):
    """gamma factor by a pressure broadening.

    Args:
       P: pressure (bar)
       T: temperature (K)
       Pself: partial pressure (bar)
       n_air: coefficient of the  temperature  dependence  of  the  air-broadened halfwidth
       gamma_air_ref: gamma air
       gamma_self_ref: gamma self

    Returns:
       gamma: pressure gamma factor (cm-1)
    """
    Patm = 1.01325  # atm (bar)
    Tref = 296.0  # reference tempearture (K)
    gamma = (Tref/T)**n_air * (gamma_air_ref *
                               ((P-Pself)/Patm) + gamma_self_ref*(Pself/Patm))
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
    return 2.6544188e-12*A


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
    return c3*jnp.sqrt(T/M)*nu_lines


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
    return c3*jnp.sqrt(T/M)*R
