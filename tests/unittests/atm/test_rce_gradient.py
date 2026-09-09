"""State-dependent convective gradients and their full-state validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce import rce_residual, solve_rce


def _flux(temperature, bottom):
    log_t = jnp.log(jnp.append(temperature, bottom))
    return jnp.concatenate(
        ((temperature[:1] / 300.0) ** 4, jnp.array([4.0, 1.0]) * jnp.diff(log_t))
    )


def _gradient(temperature, bottom):
    upper = jnp.log(temperature[0] / 300.0)
    return jnp.array(
        [0.25 + 0.05 * upper,
         0.2 + 0.1 * jnp.log(bottom / temperature[-1]) + 0.02 * upper]
    )


def _solve(**kwargs):
    inputs = dict(
        pressure_bar=np.array([1.0, 4.0]),
        pressure_boundaries_bar=np.array([0.5, 2.0, 16.0]),
        temperature_initial=np.array([250.0, 350.0]),
        bottom_temperature_initial=400.0,
        internal_flux=1.0,
        radiative_flux=_flux,
        neutral_gradient=_gradient,
        flux_atol=1.0e-9,
        flux_rtol=1.0e-9,
        gradient_atol=1.0e-9,
    )
    inputs.update(kwargs)
    return solve_rce(**inputs)


@pytest.mark.parametrize("scale", [0.6, 1.8])
@pytest.mark.parametrize("initial_mask", [[False, False], [True, True]])
def test_state_dependent_gradient_has_exact_mixed_solution(scale, initial_mask):
    result = _solve(
        temperature_initial=scale * np.array([300.0, 350.0]),
        bottom_temperature_initial=scale * 450.0,
        convective_mask_initial=np.array(initial_mask),
    )
    assert result.converged, result.status
    expected_temperature = np.array([300.0, 300.0 * np.exp(0.25)])
    lower_log_jump = 0.2 * np.log(4.0) / (1.0 - 0.1 * np.log(4.0))
    expected_bottom = expected_temperature[-1] * np.exp(lower_log_jump)
    np.testing.assert_allclose(result.temperature, expected_temperature, rtol=1e-9)
    np.testing.assert_allclose(result.bottom_temperature, expected_bottom, rtol=1e-9)
    np.testing.assert_array_equal(result.convective_mask, [False, True])
    np.testing.assert_allclose(
        result.convective_flux, [0.0, 0.0, 1.0 - lower_log_jump], atol=2e-9
    )
    current_gradient = np.asarray(_gradient(result.temperature, result.bottom_temperature))
    measured_gradient = np.diff(
        np.log(np.append(result.temperature, result.bottom_temperature))
    ) / np.log(4.0)
    np.testing.assert_allclose(
        result.gradient_residual, measured_gradient - current_gradient, atol=1e-14
    )
    assert np.max(result.gradient_residual) <= 1e-9
    assert np.max(np.abs(result.scaled_residual)) <= 1.0


def test_gradient_callback_derivatives_enter_fixed_mask_jacobian():
    log_t = jnp.log(jnp.array([300.0, 400.0, 500.0]))

    def residual(values):
        return rce_residual(
            values, jnp.array([1.0, 4.0]), 16.0, 1.0,
            _flux, _gradient, jnp.array([True, True])
        )

    matrix = jax.jit(jax.jacfwd(residual))(log_t)
    inverse_spacing = 1.0 / np.log(4.0)
    expected = np.array([
        [4.0, 0.0, 0.0],
        [-inverse_spacing - 0.05, inverse_spacing, 0.0],
        [-0.02, -inverse_spacing + 0.1, inverse_spacing - 0.1],
    ])
    np.testing.assert_allclose(matrix, expected, rtol=1e-13, atol=1e-14)
    step = 1e-5
    finite_difference = np.column_stack([
        (residual(log_t + step * direction) - residual(log_t - step * direction))
        / (2.0 * step)
        for direction in np.eye(3)
    ])
    np.testing.assert_allclose(matrix, finite_difference, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("gradient", [0.25, np.array([0.25, 0.3])])
def test_constant_callback_preserves_fixed_gradient_solution(gradient):
    fixed = _solve(neutral_gradient=gradient)
    callback = _solve(neutral_gradient=lambda t, tb: jnp.asarray(gradient))
    assert fixed.converged and callback.converged
    np.testing.assert_array_equal(callback.convective_mask, fixed.convective_mask)
    np.testing.assert_allclose(callback.temperature, fixed.temperature, rtol=1e-13)
    np.testing.assert_allclose(callback.bottom_temperature, fixed.bottom_temperature, rtol=1e-13)
    np.testing.assert_allclose(callback.gradient_residual, fixed.gradient_residual, atol=1e-14)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, 0.0, -0.1])
def test_invalid_initial_gradient_is_rejected_on_inactive_connection(value):
    with pytest.raises(ValueError, match="neutral_gradient must be finite and positive"):
        _solve(neutral_gradient=lambda t, tb: jnp.array([0.25, value]))


@pytest.mark.parametrize("shape", [(3,), (2, 1)])
def test_gradient_callback_shape_is_validated(shape):
    with pytest.raises(ValueError, match="neutral_gradient must be scalar or have shape"):
        _solve(neutral_gradient=lambda t, tb: jnp.full(shape, 0.25))


@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, 0.0, -0.1])
def test_invalid_trial_gradient_cannot_hide_on_inactive_connection(invalid):
    # The radiative root is warmer than 250 K, but only the starting state has
    # a valid gradient. No trial may bypass validation with an inactive mask.
    def gradient(temperature, bottom):
        return jnp.array([0.25, jnp.where(temperature[0] <= 250.0, 0.25, invalid)])

    result = _solve(neutral_gradient=gradient)
    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.iterations == 0
    np.testing.assert_allclose(result.temperature, [250.0, 350.0], rtol=1e-14)
    np.testing.assert_allclose(result.bottom_temperature, 400.0, rtol=1e-14)
    assert np.all(np.isfinite(result.gradient_residual))
