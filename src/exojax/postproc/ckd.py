"""Post-processing helpers for CKD spectra."""

import jax.numpy as jnp
import numpy as np

from exojax.utils.constants import c


def _wavelength_to_wavenumber(wavelength, unit):
    """Convert wavelength to wavenumber in cm-1."""
    conversion_factors = {"nm": 1.0e7, "AA": 1.0e8, "um": 1.0e4}
    try:
        return conversion_factors[unit] / wavelength
    except KeyError as exc:
        raise ValueError("unavailable unit") from exc


def wavenumber_range_with_radial_velocity(
    nu_values,
    radial_velocity_min=0.0,
    radial_velocity_max=0.0,
):
    """Return the wavenumber range covered after radial-velocity shifts.

    Args:
        nu_values: Wavenumber samples in cm-1.
        radial_velocity_min: Lower radial-velocity bound in km/s.
        radial_velocity_max: Upper radial-velocity bound in km/s.

    Returns:
        Tuple ``(nu_min, nu_max)`` covering all shifted wavenumbers.
    """
    nu_values = np.asarray(nu_values, dtype=float)
    radial_velocities = np.asarray(
        [radial_velocity_min, radial_velocity_max], dtype=float
    )
    if nu_values.size == 0:
        raise ValueError("nu_values must contain at least one element")
    if not np.all(np.isfinite(nu_values)):
        raise ValueError("nu_values must contain finite values")
    if not np.all(np.isfinite(radial_velocities)):
        raise ValueError("radial velocities must be finite")

    shift_factors = 1.0 + radial_velocities / c
    if np.any(shift_factors <= 0.0):
        raise ValueError("radial-velocity shift factors must be positive")

    shifted_nu = nu_values[:, None] * shift_factors[None, :]
    return float(np.min(shifted_nu)), float(np.max(shifted_nu))


def validate_ckd_band_coverage(nu_bands, nu_range):
    """Validate that CKD band centers cover a target wavenumber range.

    Args:
        nu_bands: CKD band-center wavenumbers in cm-1.
        nu_range: Target ``(nu_min, nu_max)`` range in cm-1.

    Raises:
        ValueError: If the CKD centers do not cover the requested range.
    """
    nu_bands = np.asarray(nu_bands, dtype=float)
    nu_range = np.asarray(nu_range, dtype=float)
    if nu_bands.size == 0:
        raise ValueError("nu_bands must contain at least one element")
    if nu_range.ndim != 1 or nu_range.size != 2:
        raise ValueError("nu_range must be a two-element sequence")
    if not np.all(np.isfinite(nu_bands)) or not np.all(np.isfinite(nu_range)):
        raise ValueError("CKD band coverage inputs must be finite")

    nu_min = float(np.min(nu_range))
    nu_max = float(np.max(nu_range))
    band_min = float(np.min(nu_bands))
    band_max = float(np.max(nu_bands))
    if band_min > nu_min or band_max < nu_max:
        raise ValueError(
            "CKD band centers do not cover the requested sampling range: "
            f"bands=[{band_min:.6g}, {band_max:.6g}] cm-1, "
            f"requested=[{nu_min:.6g}, {nu_max:.6g}] cm-1"
        )


def sample_ckd_bands_at_wavelengths(
    nu_bands,
    spectrum_bands,
    wavelength,
    radial_velocity=0.0,
    unit="nm",
):
    """Sample a CKD band spectrum at observed wavelength centers.

    Args:
        nu_bands: CKD band-center wavenumbers (cm-1).
        spectrum_bands: Spectrum values at ``nu_bands``.
        wavelength: Observed wavelength centers.
        radial_velocity: Radial velocity in km/s. The sign convention follows
            :func:`exojax.postproc.response.sampling`.
        unit: Wavelength unit, ``"nm"``, ``"AA"``, or ``"um"``.

    Returns:
        Spectrum values sampled at ``wavelength`` in the same order as the input
        wavelength array.
    """
    nu_bands = jnp.asarray(nu_bands)
    spectrum_bands = jnp.asarray(spectrum_bands)
    wavelength = jnp.asarray(wavelength)
    nu_sampling = _wavelength_to_wavenumber(wavelength, unit)
    nu_sampling_shifted = nu_sampling * (1.0 + radial_velocity / c)
    sort_index = jnp.argsort(nu_bands)
    return jnp.interp(
        nu_sampling_shifted,
        nu_bands[sort_index],
        spectrum_bands[sort_index],
    )
