"""Regression coverage for CKD interpolation values and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from exojax.opacity.ckd.core import interpolate_log_k_2d


def _reference_interpolation(table, temperatures, pressures, temperature, pressure):
    """Interpolate every column using the original two-stage definition."""
    n_t, n_p, n_g, n_bands = table.shape
    at_temperature = jax.vmap(
        lambda column: jnp.interp(temperature, temperatures, column), in_axes=1
    )(table.reshape(n_t, -1))
    at_pressure = jax.vmap(
        lambda column: jnp.interp(jnp.log(pressure), jnp.log(pressures), column),
        in_axes=1,
    )(at_temperature.reshape(n_p, -1))
    return at_pressure.reshape(n_g, n_bands)


@pytest.mark.parametrize("use_x64", [False, True])
@pytest.mark.parametrize("n_t,n_p", [(3, 3), (1, 3), (3, 1), (1, 1)])
def test_interpolation_matches_reference_values_and_gradients(use_x64, n_t, n_p):
    """Keep one-sided node slopes, clamping, and singleton behavior under JIT."""
    with enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        temperatures = jnp.array([300.0, 800.0, 1400.0], dtype=dtype)[:n_t]
        pressures = jnp.array([0.1, 1.0, 100.0], dtype=dtype)[:n_p]
        table = jnp.asarray(
            np.random.default_rng(42).normal(-40.0, 5.0, (n_t, n_p, 2, 3)),
            dtype=dtype,
        )
        t_queries, p_queries = jnp.meshgrid(
            jnp.array([200.0, 300.0, 500.0, 800.0, 1000.0, 1400.0, 1600.0]),
            jnp.array([0.01, 0.1, 0.3, 1.0, 4.0, 100.0, 200.0]),
            indexing="ij",
        )
        queries = jnp.stack((t_queries.ravel(), p_queries.ravel()), axis=1)

        def actual(query):
            return interpolate_log_k_2d(table, temperatures, pressures, *query)

        def reference(query):
            return _reference_interpolation(table, temperatures, pressures, *query)

        tolerance = 2e-6 if not use_x64 else 2e-13
        values = jax.jit(jax.vmap(actual))(queries)
        expected_values = jax.jit(jax.vmap(reference))(queries)
        assert values.dtype == expected_values.dtype == dtype
        np.testing.assert_allclose(
            values, expected_values, rtol=tolerance, atol=tolerance
        )

        derivatives = jax.jit(jax.vmap(jax.jacfwd(actual)))(queries)
        expected_derivatives = jax.jit(jax.vmap(jax.jacfwd(reference)))(queries)
        np.testing.assert_allclose(
            derivatives, expected_derivatives, rtol=tolerance, atol=tolerance
        )
        reverse_derivatives = jax.jit(
            jax.vmap(jax.grad(lambda query: jnp.sum(actual(query))))
        )(queries)
        np.testing.assert_allclose(
            reverse_derivatives,
            jnp.sum(expected_derivatives, axis=(1, 2)),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize(
    "table_dtype,grid_dtype",
    [("float32", "float64"), ("float64", "float32"), ("float32", "float32")],
)
def test_interpolation_preserves_input_precision(table_dtype, grid_dtype):
    """Coordinate and table precision remain independent as in jnp.interp."""
    with enable_x64():
        table = jnp.arange(12.0, dtype=table_dtype).reshape(2, 2, 1, 3)
        temperatures = jnp.array([300.0, 800.0], dtype=grid_dtype)
        pressures = jnp.array([0.1, 10.0], dtype=grid_dtype)
        temperature = jnp.asarray(500.0, dtype=grid_dtype)
        pressure = jnp.asarray(1.0, dtype=grid_dtype)
        arguments = (table, temperatures, pressures, temperature, pressure)
        actual = interpolate_log_k_2d(*arguments)
        expected = _reference_interpolation(*arguments)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
