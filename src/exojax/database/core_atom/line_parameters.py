"""Atomic line parameters shared by direct LPF and MODIT calculations."""

from jax import jit, vmap

from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.line_strength import line_strength
from exojax.database.core_atom.broadening import gamma_vald3


def line_parameters(
    T, PH, PHe, PHH, qr, logsij0, nu_lines, dev_nu_lines, elower, eupper,
    ielem, iion, atomicmass, ionE, gamRad, gamSta, vdWdamp, Tref,
):
    """Return line strengths and Lorentz/Doppler widths at one layer.

    Temperatures are in K, partial pressures in bar, line strengths in cm,
    and line positions and widths in cm-1. ``qr`` is Q(T)/Q(Tref), either
    per line or shared by all lines of a single species. ``dev_nu_lines``
    retains the line positions used by the atomic broadening calculation.

    Padding is handled by the caller, after calculating these parameters.
    """
    Sij = line_strength(T, logsij0, nu_lines, elower, qr, Tref)
    gammaL = gamma_vald3(
        T, PH, PHH, PHe, ielem, iion, dev_nu_lines, elower, eupper,
        atomicmass, ionE, gamRad, gamSta, vdWdamp,
    )
    sigmaD = doppler_sigma(nu_lines, T, atomicmass)
    return Sij, gammaL, sigmaD


line_parameter_matrix = jit(
    vmap(line_parameters, in_axes=(0, 0, 0, 0, 0) + (None,) * 13)
)
