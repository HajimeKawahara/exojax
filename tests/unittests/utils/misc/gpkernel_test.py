import pytest
import numpy as np
from exojax.utils.gpkernel import gpkernel_RBF
from exojax.utils.gpkernel import gpkernel_RBF_cross
from exojax.utils.gpkernel import average_covariance_gpmodel
from exojax.utils.gpkernel import average_covariance_gpmodel_cross


def test_gpkernel_RBF():
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    scale = 1.0
    amplitude = 1.0
    err = jnp.array([0.1, 0.1, 0.1])
    cov = gpkernel_RBF(x, scale, amplitude, err)
    assert np.sum(cov) == pytest.approx(5.7267933, abs=1e-5)


def test_gpkernel_RBF_cross():
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5, 4.5])
    scale = 1.0
    amplitude = 1.0
    cov = gpkernel_RBF_cross(x, y, scale, amplitude)
    assert np.sum(cov) == pytest.approx(5.801156, abs=1e-5)


def test_average_covariance_gpmodel():
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([1.1, 1.9, 3.1])
    model = jnp.array([1.0, 2.0, 3.0])
    scale = 1.0
    amplitude = 1.0
    err = jnp.array([0.1, 0.1, 0.1])
    average, covariance = average_covariance_gpmodel(
        x, data, model, scale, amplitude, err
    )
    assert np.sum(average) == pytest.approx(6.0979223, abs=1e-5)
    assert np.sum(covariance) == pytest.approx(0.059825122, abs=1e-5)


def test_average_covariance_gpmodel_cross():
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5, 4.5])
    data = jnp.array([1.1, 1.9, 3.1])
    model_x = jnp.array([1.0, 2.0, 3.0])
    model_y = jnp.array([1.5, 2.5, 3.5, 4.5])

    scale = 1.0
    amplitude = 1.0
    err_x = jnp.array([0.1, 0.1, 0.1])
    err_y = jnp.array([0.1, 0.1, 0.1, 0.1])
    average, covariance = average_covariance_gpmodel_cross(
        x, y, data, model_x, model_y, scale, amplitude, err_x, err_y
    )
    assert np.sum(average) == pytest.approx(12.213027, abs=1e-5)
    assert np.sum(covariance) == pytest.approx(1.5004411, abs=1e-5)


if __name__ == "__main__":
    test_gpkernel_RBF_cross()
    test_average_covariance_gpmodel()
    test_average_covariance_gpmodel_cross()
