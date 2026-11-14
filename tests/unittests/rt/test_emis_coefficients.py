"""Formal transfer from extinction and emissivity along a ray."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt.rtransfer import (
    rtrun_emis_pureabs_ibased_intensity_surface,
    rtrun_emis_pureabs_ray,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("alpha", [-2.0, -1e-9, 0.0, 1e-9, 1e-3, 2.0, 1e4])
def test_uniform_slab_matches_analytic_solution(dtype, alpha):
    extinction = jnp.array([[alpha]], dtype=dtype)
    emissivity = jnp.array([[1.4]], dtype=dtype)
    length = jnp.array([0.7], dtype=dtype)
    boundary = jnp.array([0.6], dtype=dtype)
    a, j, distance, bottom = (
        float(value.reshape(-1)[0])
        for value in (extinction, emissivity, length, boundary)
    )
    expected = bottom * np.exp(-a * distance)
    expected += j * distance if a == 0.0 else j / a * -np.expm1(-a * distance)

    actual = jax.jit(rtrun_emis_pureabs_ray)(
        extinction, emissivity, length, boundary
    )

    assert actual.shape == (1,)
    assert actual.dtype == dtype
    tolerance = 2e-6 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(actual, [expected], rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_zero_extinction_preserves_emission_and_analytic_gradient(dtype):
    emissivity = jnp.array([[1.5]], dtype=dtype)
    length = jnp.array([2.0], dtype=dtype)
    boundary = jnp.array([0.75], dtype=dtype)

    def intensity(alpha):
        return rtrun_emis_pureabs_ray(
            alpha.reshape((1, 1)), emissivity, length, boundary
        )[0]

    zero = jnp.asarray(0.0, dtype=dtype)
    value, derivative = jax.jit(jax.value_and_grad(intensity))(zero)
    np.testing.assert_allclose(value, 0.75 + 1.5 * 2.0)
    expected_derivative = -0.75 * 2.0 - 1.5 * 2.0**2 / 2.0
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-6)
    np.testing.assert_allclose(jax.jacfwd(intensity)(zero), expected_derivative)


def test_two_segments_and_vmapped_angles_match_formal_solution():
    extinction = jnp.array([[0.2, 0.3], [0.4, 0.8]])
    emissivity = jnp.array([[0.6, 0.9], [1.6, 0.4]])
    lengths = jnp.array([0.7, 1.2])
    boundary = jnp.array([0.5, 0.75])
    mus = jnp.array([0.3, 1.0])
    actual = jax.jit(jax.vmap(
        lambda mu: rtrun_emis_pureabs_ray(
            extinction, emissivity, lengths / mu, boundary
        )
    ))(mus)
    expected = []
    for mu in np.asarray(mus):
        transmission = np.exp(-np.asarray(extinction) * np.asarray(lengths)[:, None] / mu)
        source = np.asarray(emissivity / extinction)
        back = np.asarray(boundary) * transmission[1] + source[1] * (1 - transmission[1])
        expected.append(back * transmission[0] + source[0] * (1 - transmission[0]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    legacy = rtrun_emis_pureabs_ibased_intensity_surface(
        extinction * lengths[:, None], emissivity / extinction, boundary, mus
    )
    np.testing.assert_allclose(actual, legacy, rtol=1e-12)


def test_vacuum_and_zero_length_segments_preserve_boundary_and_emission():
    extinction = jnp.array([[0.0, 0.0], [2.0, -2.0], [0.0, 0.0]])
    emissivity = jnp.array([[0.0, 0.0], [7.0, 9.0], [1.5, 2.0]])
    # Integer path lengths and boundary exercise dtype promotion in the scan.
    lengths = jnp.array([3, 0, 2])
    boundary = jnp.array([1, 2])
    actual = rtrun_emis_pureabs_ray(extinction, emissivity, lengths, boundary)
    np.testing.assert_allclose(actual, [4.0, 6.0])


def test_integer_coefficients_are_promoted_before_path_products():
    actual = rtrun_emis_pureabs_ray(
        jnp.array([[0]], dtype=jnp.int32),
        jnp.array([[50000]], dtype=jnp.int32),
        jnp.array([50000], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    np.testing.assert_allclose(actual, [2.5e9])


@pytest.mark.parametrize("bad_input", ["emissivity", "path_length", "empty"])
def test_rejects_inconsistent_segment_shapes(bad_input):
    extinction = jnp.ones((2, 3))
    emissivity = jnp.ones((2, 3))
    lengths = jnp.ones(2)
    if bad_input == "emissivity":
        emissivity = jnp.ones(3)
    elif bad_input == "path_length":
        lengths = jnp.ones((2, 1))
    else:
        extinction, emissivity, lengths = extinction[:0], emissivity[:0], lengths[:0]
    with pytest.raises(ValueError):
        rtrun_emis_pureabs_ray(extinction, emissivity, lengths, jnp.ones(3))
