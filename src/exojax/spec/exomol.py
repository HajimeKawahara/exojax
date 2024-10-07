import numpy as np
from exojax.utils.constants import Tref_original
from exojax.utils.constants import hcperk, ccgs


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


def gamma_natural(A):
    """gamma factor by natural broadning.

    1/(4 pi c) = 2.6544188e-12 (cm-1 s)

    Args:
        A: Einstein A-factor (1/s)

    Returns:
        gamma_natural: natural width (cm-1)
    """
    return 2.6544188e-12 * A
