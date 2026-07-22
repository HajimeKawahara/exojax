"""Hydrogen integration tests for the generic bound-bound LPF kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database import AdbHydrogen
from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.broadening import gamma_natural
from exojax.opacity import OpaDirect
from exojax.opacity.bound_bound import bound_bound_absorption_emission
from exojax.opacity.lpf.lpf import voigt_profile_tensor
from exojax.opacity.lpf.make_numatrix import doppler_shifted_line_detuning
from exojax.opacity.lpf.make_numatrix import make_numatrix0
from exojax.rt.planck import piB
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity
from exojax.rt.source import source_from_opacity_emissivity
from exojax.utils.constants import hcperk


NU0 = 15232.997694444446
A_H_ALPHA = 4.4074272e7
G_LOWER = 8.0
G_UPPER = 18.0
MASS_H = 1.008


def _line_arrays():
    return (
        jnp.array([NU0]),
        jnp.array([A_H_ALPHA]),
        jnp.array([G_LOWER]),
        jnp.array([G_UPPER]),
    )


def _rest_numatrix(nu_grid):
    return make_numatrix0(
        np.asarray(nu_grid, dtype=np.float64),
        np.array([NU0], dtype=np.float64),
    )


def _profiles(numatrix0, temperatures, velocities):
    nu_lines, einstein_a, _, _ = _line_arrays()
    temperatures = jnp.asarray(temperatures)
    detuning = doppler_shifted_line_detuning(numatrix0, nu_lines, velocities)
    sigma = jax.vmap(doppler_sigma, (None, 0, None))(nu_lines, temperatures, MASS_H)
    gamma = jnp.broadcast_to(gamma_natural(einstein_a), sigma.shape)
    return voigt_profile_tensor(detuning, sigma, gamma)


def _lte_upper_population(number_density_lower, temperature):
    return (
        number_density_lower
        * (G_UPPER / G_LOWER)
        * jnp.exp(-hcperk * NU0 / temperature)
    )


def test_adb_hydrogen_halpha_hbeta_atomic_data():
    adb = AdbHydrogen(n_upper_max=4)

    vacuum_wavelength_angstrom = 1.0e8 / np.asarray(adb.nu_lines)
    np.testing.assert_allclose(
        vacuum_wavelength_angstrom,
        np.array([6564.6960635, 4862.73782481]),
        rtol=0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(adb.A),
        np.array([4.4074272e7, 8.413195e6]),
        rtol=2.0e-7,
    )


def test_opadirect_hydrogen_lte_baseline_is_finite_and_differentiable():
    adb = AdbHydrogen(nurange=(NU0 - 5.0, NU0 + 5.0))
    nu_grid = np.linspace(NU0 - 3.0, NU0 + 3.0, 2001, dtype=np.float64)
    opa = OpaDirect(adb, nu_grid)

    xs = opa.xsvector(6000.0, 1.0e-6)
    xsmatrix = opa.xsmatrix(
        jnp.array([5000.0, 7000.0]),
        jnp.array([1.0e-4, 1.0e-7]),
    )
    temperature_gradient = jax.grad(
        lambda temperature: jnp.sum(opa.xsvector(temperature, 1.0e-6))
    )(6000.0)

    assert xs.shape == (nu_grid.size,)
    assert xsmatrix.shape == (2, nu_grid.size)
    assert jnp.all(jnp.isfinite(xs))
    assert jnp.all(jnp.isfinite(xsmatrix))
    assert jnp.isfinite(temperature_gradient)


def test_halpha_lte_population_limit_gives_planck_source():
    temperature = 6500.0
    nu_grid = np.linspace(NU0 - 3.0, NU0 + 3.0, 6001, dtype=np.float64)
    profile = _profiles(
        _rest_numatrix(nu_grid), jnp.array([temperature]), jnp.array([0.0])
    )
    nu_lines, einstein_a, g_lower, g_upper = _line_arrays()
    n_lower = jnp.array([[2.0e3]])
    n_upper = _lte_upper_population(n_lower, temperature)

    alpha, eta_pi = bound_bound_absorption_emission(
        profile,
        nu_lines,
        einstein_a,
        g_lower,
        g_upper,
        n_lower,
        n_upper,
    )
    source_pi = source_from_opacity_emissivity(alpha, eta_pi)
    peak = int(jnp.argmax(profile[0, 0]))

    assert float(source_pi[0, peak]) == pytest.approx(
        float(piB(temperature, jnp.array([NU0]))[0]), rel=2.0e-6
    )


def test_halpha_kernel_is_jittable_and_has_finite_velocity_gradient():
    nu_grid = np.linspace(NU0 - 2.0, NU0 + 2.0, 2001, dtype=np.float64)
    numatrix0 = _rest_numatrix(nu_grid)
    nu_lines, einstein_a, g_lower, g_upper = _line_arrays()
    n_lower = jnp.array([[2.0e3]])
    n_upper = jnp.array([[1.0]])
    weights = jnp.linspace(0.5, 1.5, nu_grid.size)

    def objective(velocity):
        profile = _profiles(numatrix0, jnp.array([6000.0]), velocity[None])
        alpha, eta_pi = bound_bound_absorption_emission(
            profile,
            nu_lines,
            einstein_a,
            g_lower,
            g_upper,
            n_lower,
            n_upper,
        )
        return jnp.sum(weights * (alpha[0] + 1.0e-13 * eta_pi[0]))

    value, gradient = jax.jit(jax.value_and_grad(objective))(5.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def _two_layer_nlte_intensity(velocities):
    nu_grid = np.linspace(NU0 - 3.0, NU0 + 3.0, 6001, dtype=np.float64)
    temperatures = jnp.array([4500.0, 8000.0])
    profiles = _profiles(_rest_numatrix(nu_grid), temperatures, velocities)
    nu_lines, einstein_a, g_lower, g_upper = _line_arrays()

    n_lower = jnp.array([[1.0e4], [1.0e3]])
    source_temperatures = jnp.array([[3500.0], [8000.0]])
    n_upper = (
        n_lower * (G_UPPER / G_LOWER) * jnp.exp(-hcperk * NU0 / source_temperatures)
    )
    alpha, eta_pi = bound_bound_absorption_emission(
        profiles,
        nu_lines,
        einstein_a,
        g_lower,
        g_upper,
        n_lower,
        n_upper,
    )
    source_pi = source_from_opacity_emissivity(alpha, eta_pi)
    path_length = jnp.array([1.0e9, 1.0e9])
    dtau = alpha * path_length[:, None]
    intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source_pi, jnp.array([1.0]))[
        0
    ]
    return jnp.asarray(nu_grid), intensity


def test_two_layer_halpha_transfer_has_self_reversed_core():
    nu_grid, intensity = _two_layer_nlte_intensity(jnp.array([0.0, 0.0]))

    center = int(jnp.argmin(jnp.abs(nu_grid - NU0)))
    blue_peak = jnp.max(intensity[center + 1 :])
    red_peak = jnp.max(intensity[:center])
    assert intensity[center] < blue_peak
    assert intensity[center] < red_peak


def test_blueshifted_upper_halpha_absorption_suppresses_blue_peak():
    nu_grid, intensity = _two_layer_nlte_intensity(jnp.array([-5.0, 0.0]))

    center = int(jnp.argmin(jnp.abs(nu_grid - NU0)))
    blue_peak = jnp.max(intensity[center + 1 :])
    red_peak = jnp.max(intensity[:center])

    assert red_peak > blue_peak
