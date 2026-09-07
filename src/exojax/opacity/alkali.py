"""Neutral Na/K opacity with the sub-Voigt wings used by Cthulhu/POSEIDON."""

import jax.numpy as jnp
from jax import jit, vmap

from exojax.opacity.lpf.api import OpaDirect
from exojax.opacity.lpf.lpf import voigt, voigtone
from exojax.utils.constants import hcperk


@jit
def subvoigt(nuvector, sigmaD, gammaL, T, detuning_ref, wing_cutoff):
    """Evaluate one alkali profile in cm on offsets from line center (cm-1).

    The Cthulhu implementation of Baudino et al. (2015), Eq. (1), uses
    detuning_ref=30 (Na) or 20 (K) at 500 K, and wing_cutoff=5000 (Na)
    or 1600 (K), both in cm-1. sigmaD and gammaL are the Gaussian standard
    deviation and Lorentz HWHM in cm-1; T is in K. Preserve its fixed 0.998
    normalization, 9000 cm-1 truncation, and small discontinuity at detuning.
    The profile is symmetric and is not numerically renormalized.
    """
    distance = jnp.abs(nuvector)
    detuning = detuning_ref * (T / 500.0) ** 0.6
    # Keep the inactive wing branch finite at line center, including its JVP.
    wing_distance = jnp.maximum(distance, detuning)
    wing = (
        voigtone(detuning, sigmaD, gammaL)
        * (detuning / wing_distance) ** 1.5
        * jnp.exp(-hcperk * wing_distance**2 / (T * wing_cutoff))
    )
    profile = jnp.where(distance < detuning, voigt(nuvector, sigmaD, gammaL), wing)
    return jnp.where(distance <= 9000.0, profile / 0.998, 0.0)


@jit
def _xsvector(numatrix, sigmaD, gammaL, strengths, T, detuning_ref, wing_cutoff):
    profiles = vmap(subvoigt, (0, 0, 0, None, None, None))(
        numatrix, sigmaD, gammaL, T, detuning_ref, wing_cutoff
    )
    return jnp.dot(profiles.T, strengths)


class OpaAlkali(OpaDirect):
    """Convenience wrapper for OpaDirect with line_profile="alkali_subvoigt".

    Apply the Cthulhu sub-Voigt prescription to every selected line, as in
    Mullens et al. (2024), rather than only the resonance doublet. Select one
    neutral species in the database before construction and include lines up
    to 9000 cm-1 outside the evaluation grid to retain their wings.

    Line strengths, partition functions, Doppler widths, and default atomic
    broadening are inherited from OpaDirect. In particular, vmr_fraction is
    ordered H, He, H2. This matches the wing prescription, not POSEIDON's
    precomputed opacity tables or its line-specific pressure broadening.
    """

    def __init__(
        self, adb, nu_grid, wavelength_order="descending", *, atomic_broadening=None
    ):
        """Initialize with a single-species AdbVald or AdbKurucz and cm-1 grid.

        atomic_broadening optionally supplies the total Lorentz HWHM through
        the same JAX-compatible (T, P) callback as OpaDirect.
        """
        super().__init__(
            adb, nu_grid, wavelength_order, line_profile="alkali_subvoigt",
            atomic_broadening=atomic_broadening,
        )
