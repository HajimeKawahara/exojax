"""Check local Gaussian diagnostics against linear inference identities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from exojax.utils.information import linear_gaussian_diagnostics


def test_correlated_linear_gaussian_posterior():
    jacobian = np.array([[1.0, 2.0], [0.5, -1.0], [-0.5, 1.5]])
    noise_std = np.array([0.3, 0.5, 0.7])
    prior_std = np.array([0.4, 2.0])
    result = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)

    fisher = jacobian.T @ np.diag(noise_std**-2) @ jacobian
    precision = fisher + np.diag(prior_std**-2)
    covariance = np.linalg.solve(precision, np.eye(2))
    averaging_kernel = covariance @ fisher
    information_bits = (
        2.0 * np.log(prior_std).sum() - np.linalg.slogdet(covariance)[1]
    ) / (2.0 * np.log(2.0))

    assert_allclose(result["fisher"], fisher, rtol=1e-12)
    assert_allclose(result["posterior_cov"], covariance, rtol=1e-12)
    assert_allclose(result["posterior_std"], np.sqrt(np.diag(covariance)), rtol=1e-12)
    assert_allclose(result["averaging_kernel"], averaging_kernel, rtol=1e-12)
    assert_allclose(
        result["degrees_of_freedom"], np.trace(averaging_kernel), rtol=1e-12
    )
    assert_allclose(result["information_bits"], information_bits, rtol=1e-12)


def test_scalar_information_is_in_bits():
    result = linear_gaussian_diagnostics([[2.0]], [2.0], [np.sqrt(3.0)])

    assert_allclose(result["posterior_cov"], [[0.75]], rtol=1e-12)
    assert_allclose(result["averaging_kernel"], [[0.75]], rtol=1e-12)
    assert_allclose(result["degrees_of_freedom"], 0.75, rtol=1e-12)
    assert_allclose(result["information_bits"], 1.0, rtol=1e-12)


@pytest.mark.parametrize("row_weights", [(1.0,), (1.0, -2.0, 0.5, 0.0)])
def test_rank_deficient_model_preserves_unconstrained_modes(row_weights):
    row_weights = np.asarray(row_weights)
    jacobian = row_weights[:, None] * np.array([[1.0, 2.0, 0.0]])
    noise_std = np.full(len(row_weights), 2.0)
    prior_std = np.array([2.0, 3.0, 4.0])
    result = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)

    # The Sherman-Morrison formula gives the covariance for a rank-one update.
    direction = np.array([1.0, 3.0, 0.0])
    strength = row_weights @ row_weights
    expected = np.eye(3) - (
        strength * np.outer(direction, direction)
        / (1.0 + strength * (direction @ direction))
    )
    normalized_cov = result["posterior_cov"] / np.outer(prior_std, prior_std)
    assert_allclose(normalized_cov, expected, rtol=1e-12, atol=1e-12)

    singular_values = np.asarray(result["singular_values"])
    modes = np.asarray(result["parameter_modes"])
    assert singular_values.shape == (3,)
    assert modes.shape == (3, 3)
    assert_allclose(singular_values, [np.sqrt(10.0 * strength), 0.0, 0.0], atol=1e-12)
    assert_allclose(modes @ modes.T, np.eye(3), atol=1e-12)
    whitened = jacobian * prior_std[None, :] / noise_std[:, None]
    assert_allclose(whitened @ modes[1:].T, 0.0, atol=1e-12)
    assert_allclose(normalized_cov @ modes[1:].T, modes[1:].T, atol=1e-12)
    assert_allclose(result["posterior_std"][2], prior_std[2], rtol=1e-12)


def test_no_sensitivity_returns_the_prior():
    prior_std = np.array([0.1, 2.0, 5.0])
    result = linear_gaussian_diagnostics(np.zeros((2, 3)), [1.0, 2.0], prior_std)

    assert_allclose(result["posterior_cov"], np.diag(prior_std**2), atol=1e-12)
    assert_allclose(result["posterior_std"], prior_std, atol=1e-12)
    for name in (
        "fisher",
        "averaging_kernel",
        "degrees_of_freedom",
        "information_bits",
        "singular_values",
    ):
        assert_allclose(result[name], 0.0, atol=1e-12)
    modes = result["parameter_modes"]
    assert_allclose(modes @ modes.T, np.eye(3), atol=1e-12)


def test_parameter_and_observation_units_do_not_change_information():
    jacobian = np.array([[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]])
    noise_std = np.array([0.5, 1.0, 2.0])
    prior_std = np.array([2.0, 0.2])
    result = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)

    # Re-express each parameter in a different unit and the data in a new unit.
    parameter_scale = np.array([1000.0, 0.01])
    data_scale = 1e6
    rescaled = linear_gaussian_diagnostics(
        data_scale * jacobian / parameter_scale,
        data_scale * noise_std,
        parameter_scale * prior_std,
    )
    for name in ("degrees_of_freedom", "information_bits", "singular_values"):
        assert_allclose(rescaled[name], result[name], rtol=1e-12)
    assert_allclose(
        rescaled["posterior_cov"],
        result["posterior_cov"] * np.outer(parameter_scale, parameter_scale),
        rtol=1e-12,
    )
    assert_allclose(
        rescaled["posterior_std"],
        result["posterior_std"] * parameter_scale,
        rtol=1e-12,
    )
    assert_allclose(
        rescaled["averaging_kernel"],
        result["averaging_kernel"]
        * parameter_scale[:, None]
        / parameter_scale[None, :],
        rtol=1e-12,
    )


def test_independent_observation_adds_information():
    prior_std = np.array([1.0, 2.0])
    initial = linear_gaussian_diagnostics([[1.0, 1.0]], [0.5], prior_std)
    extended = linear_gaussian_diagnostics(
        [[1.0, 1.0], [1.0, -1.0]], [0.5, 0.5], prior_std
    )

    assert np.all(np.asarray(extended["posterior_std"]) < initial["posterior_std"])
    assert extended["information_bits"] > initial["information_bits"]
    assert extended["degrees_of_freedom"] > initial["degrees_of_freedom"]
    covariance_reduction = initial["posterior_cov"] - extended["posterior_cov"]
    assert np.linalg.eigvalsh(covariance_reduction).min() >= -1e-12


def test_jit_with_automatically_differentiated_model():
    def spectrum(theta):
        return jnp.array([theta[0] * theta[1], jnp.exp(theta[0]), theta[1] ** 2])

    theta = jnp.array([0.2, 0.5])
    jacobian = jax.jacfwd(spectrum)(theta)
    noise_std = jnp.array([0.1, 0.2, 0.3])
    prior_std = jnp.array([0.5, 1.0])
    eager = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)
    compiled = jax.jit(linear_gaussian_diagnostics)(jacobian, noise_std, prior_std)

    for name in eager:
        assert_allclose(compiled[name], eager[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "jacobian, noise_std, prior_std",
    [
        ([1.0, 2.0], [1.0, 1.0], [1.0]),
        (np.ones((2, 2)), [1.0], [1.0, 1.0]),
        (np.ones((2, 2)), [1.0, 1.0], [1.0]),
        (np.ones((2, 2)), [[1.0, 1.0]], [1.0, 1.0]),
        (np.ones((2, 2)), [1.0, 1.0], [[1.0, 1.0]]),
        (np.empty((0, 2)), [], [1.0, 1.0]),
        (np.empty((2, 0)), [1.0, 1.0], []),
    ],
)
def test_invalid_shapes_raise(jacobian, noise_std, prior_std):
    with pytest.raises(ValueError):
        linear_gaussian_diagnostics(jacobian, noise_std, prior_std)


@pytest.mark.parametrize("argument", ["jacobian", "noise_std", "prior_std"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_raise(argument, invalid_value):
    inputs = {
        "jacobian": np.ones((2, 2)),
        "noise_std": np.ones(2),
        "prior_std": np.ones(2),
    }
    inputs[argument].flat[0] = invalid_value
    with pytest.raises(ValueError):
        linear_gaussian_diagnostics(**inputs)


@pytest.mark.parametrize("argument", ["noise_std", "prior_std"])
@pytest.mark.parametrize("invalid_value", [0.0, -1.0])
def test_nonpositive_standard_deviations_raise(argument, invalid_value):
    inputs = {
        "jacobian": np.ones((2, 2)),
        "noise_std": np.ones(2),
        "prior_std": np.ones(2),
    }
    inputs[argument][0] = invalid_value
    with pytest.raises(ValueError):
        linear_gaussian_diagnostics(**inputs)


@pytest.mark.parametrize("argument", ["jacobian", "noise_std", "prior_std"])
def test_complex_inputs_raise(argument):
    inputs = {
        "jacobian": np.ones((2, 2)),
        "noise_std": np.ones(2),
        "prior_std": np.ones(2),
    }
    inputs[argument] = inputs[argument].astype(complex)
    with pytest.raises((TypeError, ValueError)):
        linear_gaussian_diagnostics(**inputs)
