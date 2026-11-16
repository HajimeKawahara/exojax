"""Offline NIST hydrogen profiles and population-based line coefficients."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import voigt_profile

from exojax.database import AdbNist
from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.line_strength import line_strength
from exojax.database.nist import api
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.utils.constants import c, ccgs, hcgs, hcperk


@pytest.fixture
def hydrogen(monkeypatch):
    """Retain NIST's collapsed levels and reject a redundant resolved line."""
    nist2df = pytest.importorskip("radis.api.nistapi").nist2df
    source = Path(__file__).parent / "data" / "h_i_balmer.tsv"
    # NIST ASD, retrieved 2026-11-16: H I, 3700-6600 A, vacuum, all transitions.
    # Eight n=2 -> n=3..10 rows and one resolved 2p -> 3d row, with original
    # values. See https://physics.nist.gov/PhysRefData/ASD/lines_form.html.
    frame = nist2df(source, "H_I")
    frame["species"] = "H_I"  # Added by RADIS's cache writer in production.
    monkeypatch.setattr(
        api, "fetch_nist_lines", lambda *args, **kwargs: (frame.copy(), [])
    )
    return AdbNist("H_I", nurange=[15000.0, 27000.0])


@pytest.fixture
def halpha(hydrogen):
    hydrogen.masking(hydrogen.nu_lines < 16000.0)
    grid = np.linspace(hydrogen.nu_lines[0] - 50, hydrogen.nu_lines[0] + 50, 2001)

    def broadening(T, P):
        # An explicit test model, not a damping estimate from transition A.
        return jnp.full_like(hydrogen.A, 0.001 + 0.02 * P * (8000.0 / T)**0.7)

    return hydrogen, OpaDirect(hydrogen, grid, atomic_broadening=broadening)


def _coefficients(adb, profiles, lower, upper):
    return line_coefficients(
        profiles, adb.nu_lines, adb.A, adb.glower, adb.gupper, lower, upper
    )


def test_nist_collapsed_balmer_centers_and_weights(hydrogen):
    # RADIS derives centers from the tabulated level energies, not the
    # rounded Ritz-wavelength column or a nonrelativistic Rydberg formula.
    expected_nu = np.array([
        15233.146, 20564.746, 23032.499, 24373.0101,
        25181.2928, 25705.8988, 26065.5673, 26322.8365,
    ])
    np.testing.assert_allclose(hydrogen.nu_lines, expected_nu, rtol=0, atol=2e-11)
    np.testing.assert_array_equal(hydrogen.glower, np.full(8, 8.0))
    np.testing.assert_array_equal(hydrogen.gupper, 2 * np.arange(3, 11)**2)
    np.testing.assert_allclose(hydrogen.A[0], 4.4101e7, rtol=0, atol=0)
    np.testing.assert_allclose(1e8 / hydrogen.nu_lines[0], 6564.63215149, rtol=0, atol=1e-7)
    selected = np.array([True, False, False, True, False, False, False, True])
    hydrogen.masking(selected)
    np.testing.assert_array_equal(hydrogen.gupper, [18.0, 72.0, 200.0])
    np.testing.assert_allclose(hydrogen.eupper - hydrogen.elower, expected_nu[selected])


def test_profiles_match_scipy_and_lte_opacity(hydrogen):
    hydrogen.masking(hydrogen.nu_lines < 21000.0)
    grid = np.sort(np.concatenate([
        center + np.linspace(-5.0, 5.0, 201) for center in hydrogen.nu_lines
    ]))

    def broadening(T, P):
        return jnp.array([0.01, 0.02]) + P * (8000.0 / T)**0.7

    opa = OpaDirect(hydrogen, grid, atomic_broadening=broadening)
    temperatures, pressures = jnp.array([7000.0, 11000.0]), jnp.array([0.0, 0.1])
    profiles = jax.jit(jax.vmap(opa.line_profiles))(temperatures, pressures)
    reconstructed = []
    for index, (temperature, pressure) in enumerate(zip(temperatures, pressures)):
        expected = voigt_profile(
            grid[None, :] - hydrogen.nu_lines[:, None],
            np.asarray(doppler_sigma(
                hydrogen.nu_lines, temperature, hydrogen.line_masses
            ))[:, None],
            np.asarray(broadening(temperature, pressure))[:, None],
        )
        np.testing.assert_allclose(profiles[index], expected, rtol=3e-6)
        strengths = line_strength(
            temperature, hydrogen.logsij0, hydrogen.nu_lines, hydrogen.elower,
            hydrogen.qr_interp_lines(temperature, hydrogen.Tref), hydrogen.Tref,
        )
        reconstructed.append(jnp.sum(strengths[:, None] * profiles[index], axis=0))
    np.testing.assert_allclose(
        reconstructed, opa.xsmatrix(temperatures, pressures), rtol=2e-12
    )
    np.testing.assert_allclose(opa.line_profiles(7000.0), profiles[0], rtol=2e-12)


def test_profiles_are_normalized_and_redshift_for_positive_velocity(halpha):
    adb, opa = halpha
    temperatures, velocities = jnp.array([6000.0, 10000.0]), jnp.array([-30.0, 30.0])
    profiles = jax.jit(jax.vmap(
        lambda T, v: opa.line_profiles(T, velocity=v)
    ))(temperatures, velocities)
    assert profiles.shape == (2, 1, len(opa.nu_grid))
    np.testing.assert_allclose(
        trapezoid(profiles[:, 0], opa.nu_grid, axis=-1), 1.0, atol=2e-5
    )
    peak_centers = opa.nu_grid[np.argmax(profiles[:, 0], axis=-1)]
    np.testing.assert_allclose(
        peak_centers, adb.nu_lines[0] * (1.0 - velocities / c), atol=0.025, rtol=0
    )


def test_profile_temperature_pressure_and_velocity_gradients(halpha):
    _, opa = halpha

    def signal(parameters):
        T, P, velocity = parameters
        return jnp.log(opa.line_profiles(T, P, velocity=velocity)[0, 1010])

    parameters = jnp.array([8000.0, 0.2, 5.0])
    gradient = jax.jit(jax.grad(signal))(parameters)
    steps = jnp.array([0.1, 1e-4, 1e-3])
    finite_difference = jnp.array([
        (signal(parameters + delta) - signal(parameters - delta)) / (2 * step)
        for delta, step in zip(jnp.diag(steps), steps)
    ])
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) > 1e-8)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-4, atol=1e-9)


def test_lte_populations_recover_opacity_and_planck_source(halpha):
    adb, opa = halpha
    temperatures = jnp.array([7000.0, 11000.0])
    density = jnp.array([1e12, 3e12])[:, None]
    partition = jax.vmap(
        lambda T: adb.qr_interp_lines(T, adb.Tref) * adb.QTref_284[adb._QTmask]
    )(temperatures)
    lower = density * adb.glower * jnp.exp(
        -hcperk * adb.elower / temperatures[:, None]
    ) / partition
    upper = density * adb.gupper * jnp.exp(
        -hcperk * adb.eupper / temperatures[:, None]
    ) / partition
    profiles = jax.vmap(opa.line_profiles)(temperatures)
    extinction, emissivity = _coefficients(adb, profiles, lower, upper)
    np.testing.assert_allclose(
        extinction / density, opa.xsmatrix(temperatures, jnp.zeros(2)), rtol=2e-12
    )
    expected_source = 2.0 * hcgs * ccgs**2 * adb.nu_lines[0]**3 / np.expm1(
        hcperk * adb.nu_lines[0] / temperatures
    )
    np.testing.assert_allclose(
        emissivity / extinction,
        np.broadcast_to(expected_source[:, None], extinction.shape), rtol=2e-12,
    )


@pytest.mark.parametrize("lower,upper,sign", [
    (1e6, 0.0, 1), (0.0, 0.0, 0), (8e6, 18e6, 0), (0.0, 1e6, -1),
])
def test_zero_populations_and_stimulated_emission(halpha, lower, upper, sign):
    adb, opa = halpha
    extinction, emissivity = _coefficients(
        adb, opa.line_profiles(8000.0), jnp.array([lower]), jnp.array([upper])
    )
    assert np.all(np.isfinite(extinction)) and np.all(np.isfinite(emissivity))
    assert np.all(np.sign(extinction) == sign)
    if upper == 0.0:
        np.testing.assert_array_equal(emissivity, 0.0)
    else:
        assert np.all(emissivity > 0.0)


def test_coefficient_broadcasting_requires_explicit_line_axis(halpha):
    adb, opa = halpha
    profile = opa.line_profiles(8000.0)
    lower = jnp.array([[1e6], [2e6]])
    extinction, emissivity = _coefficients(adb, profile, lower, lower * 0.1)
    assert extinction.shape == emissivity.shape == (2, len(opa.nu_grid))
    np.testing.assert_allclose(extinction[1], 2 * extinction[0], rtol=1e-12)
    np.testing.assert_allclose(emissivity[1], 2 * emissivity[0], rtol=1e-12)
    with pytest.raises(ValueError):
        _coefficients(adb, profile, lower[:, 0], lower * 0.1)


@pytest.mark.parametrize("kwargs", [
    {"T": jnp.array([8000.0])},
    {"T": 8000.0, "P": jnp.array([0.0])},
    {"T": 8000.0, "velocity": jnp.array([0.0])},
])
def test_profiles_require_scalar_layer_parameters(halpha, kwargs):
    _, opa = halpha
    with pytest.raises(ValueError, match="scalar"):
        opa.line_profiles(**kwargs)
