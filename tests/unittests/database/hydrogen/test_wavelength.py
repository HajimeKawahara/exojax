"""Hydrogen rest wavelengths and consistent effective transition energies."""

from __future__ import annotations

import numpy as np
import pytest

from exojax.database import AdbHydrogen
from exojax.opacity import OpaDirect
from exojax.rt.planck import piB
from exojax.utils.constants import ccgs, hcgs, hcperk


@pytest.mark.parametrize("mode,wavelength", [
    ("reference", 6564.614),
    ("rydberg", 6564.6960634984225),
])
def test_opacity_peak_on_a_fixed_vacuum_wavelength_grid(mode, wavelength):
    adb = AdbHydrogen(n_upper_max=3, wavelength_model=mode)
    wavelengths = np.linspace(6564.2, 6565.1, 4501)
    grid = 1e8 / wavelengths[::-1]
    spectrum = np.asarray(OpaDirect(adb, grid).xsvector(8000.0, 0.0))
    assert np.isfinite(spectrum).all() and spectrum.max() > 0
    peak_wavelength = 1e8 / grid[np.argmax(spectrum)]
    np.testing.assert_allclose(peak_wavelength, wavelength, rtol=0, atol=3e-4)


@pytest.mark.parametrize("gpu_transfer", [False, True])
@pytest.mark.parametrize("mode,wavelength", [
    ("reference", 6564.614),
    ("rydberg", 6564.6960634984225),
])
def test_wavelength_is_applied_before_narrow_line_selection(mode, wavelength, gpu_transfer):
    center = 1e8 / wavelength
    adb = AdbHydrogen(
        nurange=[center - 0.02, center + 0.02],
        wavelength_model=mode, gpu_transfer=gpu_transfer,
    )
    np.testing.assert_array_equal(adb.n_upper, [3])
    np.testing.assert_allclose(adb.nu_lines, [center], rtol=1e-14)
    if not gpu_transfer:
        adb.generate_jnp_arrays()
    np.testing.assert_allclose(adb.dev_nu_lines, [center], rtol=1e-14)


def test_reference_default_changes_only_halpha_and_preserves_transition_consistency():
    reference = AdbHydrogen()
    rydberg = AdbHydrogen(wavelength_model="rydberg")
    np.testing.assert_allclose(1e8 / reference.nu_lines[0], 6564.614, rtol=1e-14)
    np.testing.assert_allclose(1e8 / rydberg.nu_lines[0], 6564.6960634984225, rtol=1e-14)
    for field in ("nu_lines", "_eupper", "_A", "Sij0"):
        np.testing.assert_array_equal(getattr(reference, field)[1:], getattr(rydberg, field)[1:])
    np.testing.assert_array_equal(reference._elower, rydberg._elower)
    np.testing.assert_allclose(
        reference._A[0] / rydberg._A[0], (6564.6960634984225 / 6564.614)**2,
        rtol=1e-14,
    )
    for adb in (reference, rydberg):
        np.testing.assert_allclose(adb._eupper - adb._elower, adb.nu_lines, rtol=2e-15)
        temperature = 8000.0
        population_ratio = (18.0 / 8.0) * np.exp(
            -hcperk * (adb._eupper[0] - adb._elower[0]) / temperature
        )
        source = (
            2 * hcgs * ccgs**2 * adb.nu_lines[0]**3
            / ((18.0 / 8.0) / population_ratio - 1)
        )
        planck = np.asarray(piB(temperature, np.asarray([adb.nu_lines[0]])))[0] / np.pi
        # piB's stored prefactor differs from 2*h*c**2*pi by 1.7e-8.
        np.testing.assert_allclose(source, planck, rtol=2e-8)


def test_invalid_wavelength_model_is_rejected():
    with pytest.raises(ValueError, match="wavelength_model"):
        AdbHydrogen(wavelength_model="air")
