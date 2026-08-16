import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt.planck import fac_planck, piB, piBarr
from exojax.utils.constants import hcperk


@pytest.mark.parametrize(
    "planck_function",
    [
        lambda temperature, nu_grid: piB(temperature, nu_grid),
        lambda temperature, nu_grid: piBarr(
            jnp.atleast_1d(temperature), nu_grid
        )[0],
    ],
    ids=["piB", "piBarr"],
)
def test_float32_forward_and_gradients(planck_function):
    temperature = jnp.float32(100.0)
    nu_grid = jnp.array(
        [44.0 * float(temperature) / hcperk, 6200.0], dtype=jnp.float32
    )

    exponent = hcperk * np.asarray(nu_grid, dtype=np.float64) / float(temperature)
    expected = fac_planck * np.asarray(nu_grid, dtype=np.float64) ** 3 / np.expm1(
        exponent
    )
    expected_temperature_gradient = (
        expected
        * exponent
        / float(temperature)
        / (-np.expm1(-exponent))
    )
    expected_wavenumber_gradient = expected * (
        3.0 / np.asarray(nu_grid, dtype=np.float64)
        - (hcperk / float(temperature)) / (-np.expm1(-exponent))
    )

    actual = planck_function(temperature, nu_grid)
    actual_temperature_gradient = jax.jacrev(
        lambda value: planck_function(value, nu_grid)
    )(temperature)
    actual_wavenumber_gradient = jnp.diag(
        jax.jacrev(lambda value: planck_function(temperature, value))(nu_grid)
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=0.0)
    np.testing.assert_allclose(
        actual_temperature_gradient,
        expected_temperature_gradient,
        rtol=1.0e-5,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual_wavenumber_gradient,
        expected_wavenumber_gradient,
        rtol=1.0e-5,
        atol=0.0,
    )
