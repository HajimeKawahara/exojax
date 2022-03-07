import numpy as np
from exojax.utils.constants import hcperk, ccgs


def Sij0(A, g, nu_lines, elower, QTref):
    """Reference Line Strength in Tref=296K, S0.

    Note:
       Tref=296K

    Args:
       A: Einstein coefficient (s-1)
       g: the upper state statistical weight
       nu_lines: line center wavenumber (cm-1)
       elower: elower
       QTref: partition function Q(Tref)
       Mmol: molecular mass (normalized by m_u)

    Returns:
       Sij(T): Line strength (cm)
    """
    Tref = 296.0
    S0 = -A*g*np.exp(-hcperk*elower/Tref)*np.expm1(-hcperk*nu_lines/Tref)\
        / (8.0*np.pi*ccgs*nu_lines**2*QTref)
    return S0


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
    Tref = 296.0  # reference tempearture (K)
    gamma = alpha_ref*P*(Tref/T)**n_air
    return gamma


def gamma_natural(A):
    """gamma factor by natural broadning.

    1/(4 pi c) = 2.6544188e-12 (cm-1 s)

    Args:
       A: Einstein A-factor (1/s)

    Returns:
       gamma_natural: natural width (cm-1)
    """
    return 2.6544188e-12*A
