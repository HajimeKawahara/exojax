"""Hydrogen profiles, level populations, and their connection to transfer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import voigt_profile

from exojax.database import AdbHydrogen
from exojax.database.core.broadening import doppler_sigma, gamma_natural
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.rt.layeropacity import layer_optical_depth_from_extinction
from exojax.rt.planck import piB
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity_surface
from exojax.utils.constants import c, ccgs, hcgs, hcperk


@pytest.fixture
def halpha():
    adb = AdbHydrogen(n_upper_max=3)
    grid = np.linspace(adb.nu_lines[0] - 50.0, adb.nu_lines[0] + 50.0, 2001)
    return adb, OpaDirect(adb, grid)


def _coefficients(adb, profiles, lower, upper):
    return line_coefficients(
        profiles, adb.nu_lines, adb.A, adb.glower, adb.gupper, lower, upper
    )


def test_profiles_and_lte_cross_sections_match_scipy():
    adb = AdbHydrogen(n_upper_max=4)
    grid = np.sort(np.concatenate([
        center + np.linspace(-5.0, 5.0, 201) for center in adb.nu_lines
    ]))
    opa = OpaDirect(adb, grid)
    temperatures = jnp.array([7000.0, 11000.0])
    expected = []
    for temperature in temperatures:
        profiles = voigt_profile(
            grid[None, :] - adb.nu_lines[:, None],
            np.asarray(doppler_sigma(adb.nu_lines, temperature, adb.line_masses))[:, None],
            np.asarray(gamma_natural(adb.A))[:, None],
        )
        strengths = (
            adb.A * adb.gupper * np.exp(-hcperk * adb.elower / temperature)
            * -np.expm1(-hcperk * adb.nu_lines / temperature)
            / (8.0 * np.pi * ccgs * adb.nu_lines**2 * adb.QT_interp("H 1", temperature))
        )
        expected.append(np.sum(np.asarray(strengths)[:, None] * profiles, axis=0))
        np.testing.assert_allclose(opa.line_profiles(temperature), profiles, rtol=3e-6)
    actual = jax.jit(opa.xsmatrix)(temperatures, jnp.array([0.0, 1.0]))
    np.testing.assert_allclose(actual, expected, rtol=3e-6)
    for index, temperature in enumerate(temperatures):
        np.testing.assert_allclose(opa.xsvector(temperature, 0.1), actual[index], rtol=1e-12)


def test_layer_profiles_are_normalized_and_redshift_for_positive_velocity(halpha):
    adb, opa = halpha
    temperatures, velocities = jnp.array([6000.0, 10000.0]), jnp.array([-30.0, 30.0])
    profiles = jax.jit(jax.vmap(opa.line_profiles))(temperatures, velocities)
    assert profiles.shape == (2, 1, len(opa.nu_grid))
    np.testing.assert_allclose(trapezoid(profiles[:, 0], opa.nu_grid, axis=-1), 1.0, atol=2e-6)
    peak_centers = opa.nu_grid[np.argmax(profiles[:, 0], axis=-1)]
    np.testing.assert_allclose(
        peak_centers, adb.nu_lines[0] * (1.0 - velocities / c), atol=0.025, rtol=0.0
    )
    for index in range(2):
        np.testing.assert_allclose(
            profiles[index], opa.line_profiles(temperatures[index], velocities[index]), rtol=1e-12
        )


def test_lte_populations_recover_opacity_and_planck_source(halpha):
    adb, opa = halpha
    temperatures = jnp.array([7000.0, 11000.0])
    density = jnp.array([1e12, 3e12])[:, None]
    partition = jax.vmap(lambda t: adb.QT_interp("H 1", t))(temperatures)[:, None]
    lower = (
        density * adb.glower
        * jnp.exp(-hcperk * adb.elower / temperatures[:, None]) / partition
    )
    upper = (
        density * adb.gupper
        * jnp.exp(-hcperk * adb.eupper / temperatures[:, None]) / partition
    )
    profiles = jax.vmap(opa.line_profiles)(temperatures)
    extinction, emissivity = jax.jit(_coefficients, static_argnums=0)(
        adb, profiles, lower, upper
    )
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


@pytest.mark.parametrize("lower,upper,sign", [(1e6, 0.0, 1), (0.0, 0.0, 0), (0.0, 1e6, -1)])
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


def test_statistical_weights_follow_line_selection():
    adb = AdbHydrogen(gpu_transfer=False)
    selected = np.array([False, True, False, False, True, False, False, False])
    adb.masking(selected)
    adb.generate_jnp_arrays()
    np.testing.assert_array_equal(adb.glower, 2 * adb.n_lower**2)
    np.testing.assert_array_equal(adb.gupper, 2 * adb.n_upper**2)
    np.testing.assert_array_equal(adb.glower, adb._glower)
    assert adb.glower.shape == adb.A.shape == (2,)


def test_coefficient_broadcasting_requires_explicit_line_axis(halpha):
    adb, opa = halpha
    profile = opa.line_profiles(8000.0)
    lower = jnp.array([[1e6], [2e6]])
    extinction, emissivity = _coefficients(adb, profile, lower, lower * 0.1)
    assert extinction.shape == emissivity.shape == (2, len(opa.nu_grid))
    np.testing.assert_allclose(extinction[1], 2 * extinction[0], rtol=1e-12)
    np.testing.assert_allclose(emissivity[1], 2 * emissivity[0], rtol=1e-12)
    extinction, emissivity = _coefficients(adb, profile, lower, jnp.array([1e5]))
    assert extinction.shape == emissivity.shape == (2, len(opa.nu_grid))
    np.testing.assert_array_equal(emissivity[0], emissivity[1])
    with pytest.raises(ValueError):
        _coefficients(adb, profile, lower[:, 0], lower * 0.1)
    with pytest.raises(ValueError):
        line_coefficients(
            profile, jnp.ones(2), adb.A, adb.glower, adb.gupper, lower, lower
        )


def test_two_moving_layers_match_formal_solution_and_have_gradients(halpha):
    adb, opa = halpha
    mus = jnp.array([0.4, 1.0])
    boundary = piB(4000.0, opa.nu_grid) / jnp.pi

    def transfer(parameters):
        temperatures = parameters[0] * jnp.array([0.8, 1.2])
        lower = jnp.exp(parameters[1]) * jnp.array([[0.4], [1.0]])
        upper = lower * jnp.array([[0.02], [0.15]])
        velocities = parameters[2] * jnp.array([-1.0, 0.5])
        profiles = jax.vmap(opa.line_profiles)(temperatures, velocities)
        extinction, emissivity = _coefficients(adb, profiles, lower, upper)
        dtau = layer_optical_depth_from_extinction(extinction, jnp.array([2e7, 1e7]))
        source = emissivity / extinction
        intensity = rtrun_emis_pureabs_ibased_intensity_surface(dtau, source, boundary, mus)
        transmission = jnp.exp(-dtau[None, :, :] / mus[:, None, None])
        expected = (
            source[0] * (1 - transmission[:, 0])
            + transmission[:, 0] * source[1] * (1 - transmission[:, 1])
            + boundary * transmission[:, 0] * transmission[:, 1]
        )
        return intensity, expected

    parameters = jnp.array([8000.0, np.log(1e6), 5.0])
    actual, expected = jax.jit(transfer)(parameters)
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    objective = lambda p: jnp.log(transfer(p)[0][0, 1010])
    gradient = jax.jit(jax.grad(objective))(parameters)
    steps = jnp.array([0.2, 1e-4, 1e-3])
    differences = jnp.diag(steps)
    finite_difference = jnp.array([
        (objective(parameters + delta) - objective(parameters - delta)) / (2 * step)
        for delta, step in zip(differences, steps)
    ])
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) > 1e-8)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-4, atol=1e-8)
