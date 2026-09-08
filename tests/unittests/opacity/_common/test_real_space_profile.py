"""Numerical checks against independent, discrete Voigt convolutions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import voigt_profile

from exojax.opacity._common.profconv import calc_xsection_from_lsd_zeroscan


def _density(length=32):
    density = np.zeros((length, 2))
    density[[0, 7, length - 1], 0] = [0.4, 1.3, 0.2]
    density[[0, 18, length - 1], 1] = [0.3, 0.5, 0.7]
    return density


def _direct_convolution(density, sigma, gammas, resolution, nu_grid):
    coordinates = np.arange(len(density))
    offsets = coordinates[:, None] - coordinates[None, :]
    profiles = voigt_profile(offsets[:, :, None], sigma, gammas)
    return np.sum(profiles * density[None, :, :], axis=(1, 2)) * resolution / nu_grid


def _convolve(density, sigma, gammas, resolution, nu_grid, **kwargs):
    pmarray = (-1.0) ** jnp.arange(len(density) + 1)
    return calc_xsection_from_lsd_zeroscan(
        jnp.asarray(density),
        resolution,
        pmarray,
        sigma,
        jnp.asarray(nu_grid),
        jnp.log(gammas),
        **kwargs,
    )


@pytest.mark.parametrize("sigma", [0.3, 1.2])
def test_real_space_matches_discrete_voigt_at_both_boundaries(sigma):
    density = _density()
    gammas = jnp.array([0.02, 1.4])
    nu_grid = np.geomspace(2000.0, 2010.0, len(density))
    actual = jax.jit(partial(_convolve, profile_kernel="real_space"))(
        density, sigma, gammas, 2400.0, nu_grid
    )
    expected = _direct_convolution(density, sigma, gammas, 2400.0, nu_grid)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-9)
    assert np.min(actual) >= 0.0


def test_real_space_temperature_and_width_jvp_vjp_match_scipy_differences():
    density = _density()
    nu_grid = np.geomspace(2000.0, 2010.0, len(density))

    def spectrum(parameters):
        temperature, gaussian_scale, lorentz_scale = parameters
        sigma = gaussian_scale * jnp.sqrt(temperature)
        gammas = (
            jnp.array([0.05, 0.4])
            * lorentz_scale
            * temperature ** -jnp.array([0.4, 0.7])
        )
        return _convolve(
            density, sigma, gammas, 2400.0, nu_grid, profile_kernel="real_space"
        )

    def reference(parameters):
        temperature, gaussian_scale, lorentz_scale = parameters
        sigma = gaussian_scale * np.sqrt(temperature)
        gammas = (
            np.array([0.05, 0.4])
            * lorentz_scale
            * temperature ** -np.array([0.4, 0.7])
        )
        return _direct_convolution(density, sigma, gammas, 2400.0, nu_grid)

    position = jnp.array([1.1, 0.6, 0.7])
    direction = jnp.array([0.3, -0.2, 0.4])
    step = 1e-5
    _, tangent = jax.jvp(jax.jit(spectrum), (position,), (direction,))
    finite_tangent = (
        reference(position + step * direction) - reference(position - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite_tangent, rtol=2e-5, atol=2e-8)

    cotangent = np.linspace(-0.4, 0.8, len(density))
    _, pullback = jax.vjp(jax.jit(spectrum), position)
    (gradient,) = pullback(jnp.asarray(cotangent))
    finite_gradient = np.array(
        [
            cotangent
            @ (reference(position + step * axis) - reference(position - step * axis))
            / (2 * step)
            for axis in np.eye(len(position))
        ]
    )
    np.testing.assert_allclose(gradient, finite_gradient, rtol=2e-5, atol=2e-8)


def test_analytic_default_is_unchanged():
    density = _density()
    args = (density, 0.6, jnp.array([0.05, 0.4]), 1.0, jnp.ones(len(density)))
    np.testing.assert_array_equal(
        _convolve(*args), _convolve(*args, profile_kernel="analytic")
    )


def test_real_space_avoids_negative_kernel_ringing():
    density = jnp.zeros((32, 1)).at[16, 0].set(1.0)
    args = (density, 0.3, jnp.array([1e-6]), 1.0, jnp.ones(32))
    analytic = _convolve(*args, profile_kernel="analytic")
    real_space = _convolve(*args, profile_kernel="real_space")
    expected = voigt_profile(np.arange(32) - 16, 0.3, 1e-6)
    assert np.min(analytic) < -0.01
    assert np.min(real_space) >= 0.0
    np.testing.assert_allclose(real_space, expected, rtol=2e-6, atol=5e-9)


def test_invalid_profile_kernel_is_rejected():
    density = _density()
    with pytest.raises(ValueError, match="profile_kernel"):
        _convolve(
            density, 0.6, jnp.array([0.05, 0.4]), 1.0, jnp.ones(32),
            profile_kernel="invalid",
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_real_space_float32_inputs_preserve_precision(enable_x64):
    with jax.experimental.enable_x64(enable_x64):
        density = jnp.asarray(_density(), dtype=jnp.float32)
        sigma = jnp.float32(0.6)
        gammas = jnp.array([0.05, 0.4], dtype=jnp.float32)
        nu_grid = jnp.ones(len(density), dtype=jnp.float32)
        actual = _convolve(
            density,
            sigma,
            gammas,
            jnp.float32(1.0),
            nu_grid,
            profile_kernel="real_space",
        )
        expected = _direct_convolution(density, sigma, gammas, 1.0, nu_grid)
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-7)
