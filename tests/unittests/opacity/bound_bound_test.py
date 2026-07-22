"""Tests for generic bound-bound radiative coefficients."""

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.bound_bound import bound_bound_absorption_emission
from exojax.opacity.bound_bound import population_inversion_mask


NU0 = 15232.997694444446
A_VALUE = 4.4074272e7
G_LOWER = 8.0
G_UPPER = 18.0


def test_absorption_and_emissivity_match_integrated_coefficients():
    unit_profile = jnp.ones((1, 1, 1))
    n_lower = jnp.array([[2.0e3]])
    n_upper = jnp.array([[3.0]])

    alpha, eta_pi = bound_bound_absorption_emission(
        unit_profile,
        jnp.array([NU0]),
        jnp.array([A_VALUE]),
        jnp.array([G_LOWER]),
        jnp.array([G_UPPER]),
        n_lower,
        n_upper,
    )

    expected_alpha = (
        A_VALUE
        / (8.0 * np.pi * 2.99792458e10 * NU0**2)
        * (G_UPPER / G_LOWER * 2.0e3 - 3.0)
    )
    expected_eta = 6.62607015e-27 * 2.99792458e10 * NU0 / 4.0 * A_VALUE * 3.0
    assert float(alpha[0, 0]) == pytest.approx(expected_alpha, rel=2.0e-6)
    assert float(eta_pi[0, 0]) == pytest.approx(expected_eta, rel=2.0e-6)


def test_population_inversion_is_identified():
    n_lower = jnp.array([[1.0], [1.0]])
    n_upper = jnp.array([[2.0], [3.0]])
    mask = population_inversion_mask(
        n_lower,
        n_upper,
        jnp.array([G_LOWER]),
        jnp.array([G_UPPER]),
    )

    np.testing.assert_array_equal(mask, np.array([[False], [True]]))


def test_population_shape_must_match_profile_layers_and_lines():
    with pytest.raises(ValueError, match="number_density_upper must have shape"):
        bound_bound_absorption_emission(
            jnp.ones((2, 1, 3)),
            jnp.array([NU0]),
            jnp.array([A_VALUE]),
            jnp.array([G_LOWER]),
            jnp.array([G_UPPER]),
            jnp.ones((2, 1)),
            jnp.ones((1, 1)),
        )
