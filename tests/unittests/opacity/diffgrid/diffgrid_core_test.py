import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.diffgrid.core import cross_section_matrix
from exojax.opacity.diffgrid.core import cubic_hermite_interpolation


def _cubic(x):
    return 0.7 * x**3 - 1.2 * x**2 + 0.4 * x + 2.0


def _cubic_derivative(x):
    return 2.1 * x**2 - 2.4 * x + 0.4


def test_cubic_hermite_is_continuous_at_internal_node():
    x0 = jnp.asarray(0.4)
    x1 = jnp.asarray(1.1)
    x2 = jnp.asarray(2.0)

    def left(x):
        return cubic_hermite_interpolation(
            x,
            x0,
            x1,
            _cubic(x0),
            _cubic(x1),
            _cubic_derivative(x0),
            _cubic_derivative(x1),
        )

    def right(x):
        return cubic_hermite_interpolation(
            x,
            x1,
            x2,
            _cubic(x1),
            _cubic(x2),
            _cubic_derivative(x1),
            _cubic_derivative(x2),
        )

    np.testing.assert_allclose(left(x1), right(x1), rtol=2.0e-6)
    np.testing.assert_allclose(left(x1), _cubic(x1), rtol=2.0e-6)
    np.testing.assert_allclose(
        jax.grad(left)(x1), jax.grad(right)(x1), rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(
        jax.grad(left)(x1), _cubic_derivative(x1), rtol=2.0e-6, atol=2.0e-6
    )

    temperature_node = 1.0 / x1

    def left_temperature(temperature):
        return left(1.0 / temperature)

    def right_temperature(temperature):
        return right(1.0 / temperature)

    np.testing.assert_allclose(
        jax.grad(left_temperature)(temperature_node),
        jax.grad(right_temperature)(temperature_node),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_cross_section_matrix_clips_logarithms_to_finite_range():
    inverse_temperature_grid = jnp.asarray([5.0e-4, 2.0e-3])
    log_cross_section_grid = jnp.asarray(
        [[[-1.0e6, 1.0e6], [-1.0e6, 1.0e6]]]
    )
    derivative_grid = jnp.zeros_like(log_cross_section_grid)
    log_floor = jnp.log(jnp.asarray(1.0e-30))

    xsmatrix = cross_section_matrix(
        jnp.asarray([1000.0]),
        inverse_temperature_grid,
        log_cross_section_grid,
        derivative_grid,
        log_floor,
    )

    assert np.all(np.isfinite(np.asarray(xsmatrix)))
    assert np.all(np.asarray(xsmatrix) > 0.0)


@pytest.mark.parametrize("temperature", [0.0, -100.0, 400.0, 2500.0, np.nan])
def test_cross_section_matrix_marks_invalid_temperature(temperature):
    inverse_temperature_grid = jnp.asarray([5.0e-4, 2.0e-3])
    log_cross_section_grid = jnp.zeros((1, 2, 3))
    derivative_grid = jnp.zeros_like(log_cross_section_grid)

    xsmatrix = cross_section_matrix(
        jnp.asarray([temperature]),
        inverse_temperature_grid,
        log_cross_section_grid,
        derivative_grid,
        jnp.asarray(-80.0),
    )

    assert np.all(np.isnan(np.asarray(xsmatrix)))
