"""Hydrogen rest wavelengths and consistent effective transition energies."""

from __future__ import annotations

import numpy as np
import pytest

from exojax.database import AdbHydrogen
from exojax.opacity import OpaDirect
from exojax.rt.planck import piB
from exojax.utils.constants import ccgs, hcgs, hcperk


# Reference values: SDSS H-alpha and NIST ASD unresolved n=2 -> n Ritz lines.
# Rydberg values preserve the pre-correction calculation for regression tests.
BALMER_WAVELENGTHS = [
    (3, 6564.614, 6564.6960634984225),
    (4, 4862.691, 4862.7378248136465),
    (5, 4341.691, 4341.73020072647),
    (6, 4102.8991, 4102.935039686515),
    (7, 3971.2020, 3971.2358902644783),
    (8, 3890.1577, 3890.190259850917),
    (9, 3836.4789, 3836.510686460117),
    (10, 3798.9827, 3799.0139256356615),
]


@pytest.mark.parametrize("mode", ["reference", "rydberg"])
@pytest.mark.parametrize("upper,reference,legacy", BALMER_WAVELENGTHS)
def test_opacity_peak_on_a_fixed_vacuum_wavelength_grid(mode, upper, reference, legacy):
    adb = AdbHydrogen(n_upper_min=upper, n_upper_max=upper, wavelength_model=mode)
    wavelengths = np.linspace(reference - 0.5, reference + 0.5, 5001)
    grid = 1e8 / wavelengths[::-1]
    spectrum = np.asarray(OpaDirect(adb, grid).xsvector(8000.0, 0.0))
    assert np.isfinite(spectrum).all() and spectrum.max() > 0
    peak_wavelength = 1e8 / grid[np.argmax(spectrum)]
    wavelength = reference if mode == "reference" else legacy
    np.testing.assert_allclose(peak_wavelength, wavelength, rtol=0, atol=3e-4)


@pytest.mark.parametrize("gpu_transfer", [False, True])
@pytest.mark.parametrize("mode", ["reference", "rydberg"])
@pytest.mark.parametrize("upper,reference,legacy", [BALMER_WAVELENGTHS[0], BALMER_WAVELENGTHS[-1]])
def test_wavelength_is_applied_before_narrow_line_selection(
    mode, upper, reference, legacy, gpu_transfer,
):
    wavelength = reference if mode == "reference" else legacy
    center = 1e8 / wavelength
    adb = AdbHydrogen(
        nurange=[center - 0.02, center + 0.02],
        wavelength_model=mode, gpu_transfer=gpu_transfer,
    )
    np.testing.assert_array_equal(adb.n_upper, [upper])
    np.testing.assert_allclose(adb.nu_lines, [center], rtol=1e-14)
    if not gpu_transfer:
        adb.generate_jnp_arrays()
    np.testing.assert_allclose(adb.dev_nu_lines, [center], rtol=1e-14)


def test_reference_default_preserves_transition_consistency_for_all_balmer_lines():
    reference = AdbHydrogen()
    rydberg = AdbHydrogen(wavelength_model="rydberg")
    expected_reference = np.array([row[1] for row in BALMER_WAVELENGTHS])
    expected_legacy = np.array([row[2] for row in BALMER_WAVELENGTHS])
    np.testing.assert_allclose(1e8 / reference.nu_lines, expected_reference, rtol=1e-14)
    np.testing.assert_allclose(1e8 / rydberg.nu_lines, expected_legacy, rtol=1e-14)
    np.testing.assert_array_equal(rydberg._eupper, [
        97491.18524444444, 102822.73443750001, 105290.480064, 106630.98386111112,
        107439.26537142857, 107963.871159375, 108323.53916049382, 108580.807566,
    ])
    np.testing.assert_array_equal(reference._elower, rydberg._elower)
    np.testing.assert_allclose(
        reference._A / rydberg._A, (expected_legacy / expected_reference)**2,
        rtol=1e-14,
    )
    for adb in (reference, rydberg):
        np.testing.assert_allclose(adb._eupper - adb._elower, adb.nu_lines, rtol=2e-15)
        temperature = 8000.0
        weight_ratio = adb._gupper / 8.0
        population_ratio = weight_ratio * np.exp(
            -hcperk * (adb._eupper - adb._elower) / temperature
        )
        source = (
            2 * hcgs * ccgs**2 * adb.nu_lines**3
            / (weight_ratio / population_ratio - 1)
        )
        planck = np.asarray(piB(temperature, adb.nu_lines)) / np.pi
        # piB's stored prefactor differs from 2*h*c**2*pi by 1.7e-8.
        np.testing.assert_allclose(source, planck, rtol=2e-8)


def test_invalid_wavelength_model_is_rejected():
    with pytest.raises(ValueError, match="wavelength_model"):
        AdbHydrogen(wavelength_model="air")
