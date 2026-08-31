"""Atomic transition parameter conversions."""

import numpy as np

from exojax.utils.constants import ccgs
from exojax.utils.constants import ecgs
from exojax.utils.constants import mecgs


def einstein_a_from_loggf(nu_lines, loggf, gupper):
    """Convert log gf values to Einstein A coefficients.

    Args:
        nu_lines: Line-center wavenumbers in cm-1.
        loggf: Log10 of lower-state statistical weight times oscillator strength.
        gupper: Upper-state statistical weights.

    Returns:
        Einstein A coefficients in s-1.
    """
    return (
        10**loggf
        / gupper
        * (ccgs * nu_lines) ** 2
        * (8.0 * np.pi**2 * ecgs**2)
        / (mecgs * ccgs**3)
    )


def einstein_a_from_oscillator_strength(nu_lines, f_lu, glower, gupper):
    """Convert absorption oscillator strengths to Einstein A coefficients.

    Args:
        nu_lines: Line-center wavenumbers in cm-1.
        f_lu: Absorption oscillator strengths.
        glower: Lower-state statistical weights.
        gupper: Upper-state statistical weights.

    Returns:
        Einstein A coefficients in s-1.
    """
    return einstein_a_from_loggf(nu_lines, np.log10(glower * f_lu), gupper)
