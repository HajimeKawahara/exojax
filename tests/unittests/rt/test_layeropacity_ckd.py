import jax.numpy as jnp
import numpy as np

from exojax.rt.layeropacity import (
    layer_optical_depth_ckd,
    layer_optical_depth_from_cross_section,
)
from exojax.utils.constants import opacity_factor


def test_layer_optical_depth_ckd_accepts_profile_mass_and_gravity():
    dpressure = np.array([0.1, 0.2])
    xstensor_ckd = np.ones((2, 3, 4))
    mixing_ratio = np.array([1.0e-3, 2.0e-3])
    mass = np.array([2.3, 2.4])
    gravity = np.array([900.0, 1000.0])

    dtau = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )

    expected_layer = opacity_factor * dpressure * mixing_ratio / (mass * gravity)
    expected = expected_layer[:, None, None] * np.ones_like(xstensor_ckd)
    np.testing.assert_allclose(np.asarray(dtau), expected)


def test_layer_optical_depth_ckd_accepts_column_layer_profiles():
    dpressure = np.array([[0.1], [0.2]])
    xstensor_ckd = np.ones((2, 3, 4))
    mixing_ratio = np.array([[1.0e-3], [2.0e-3]])
    mass = np.array([[2.3], [2.4]])
    gravity = np.array([[900.0], [1000.0]])

    dtau = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )

    expected_layer = (
        opacity_factor
        * dpressure[:, 0]
        * mixing_ratio[:, 0]
        / (mass[:, 0] * gravity[:, 0])
    )
    expected = expected_layer[:, None, None] * np.ones_like(xstensor_ckd)
    np.testing.assert_allclose(np.asarray(dtau), expected)


def test_layer_optical_depth_ckd_accepts_scalar_factors():
    dpressure = np.array([0.1, 0.2])
    xstensor_ckd = np.ones((2, 3, 4))
    mixing_ratio = 1.0e-3
    mass = 18.0
    gravity = 950.0

    dtau = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )

    expected_layer = opacity_factor * dpressure * mixing_ratio / (mass * gravity)
    expected = expected_layer[:, None, None] * np.ones_like(xstensor_ckd)
    np.testing.assert_allclose(np.asarray(dtau), expected)


def test_layer_optical_depth_ckd_accepts_prebroadcast_factors():
    dpressure = np.array([0.1, 0.2])
    xstensor_ckd = np.ones((2, 3, 4))
    mixing_ratio = np.full((2, 3, 4), 1.0e-3)
    mass = 18.0
    gravity = np.array([900.0, 1000.0])

    dtau = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )

    expected_layer = opacity_factor * dpressure / (mass * gravity)
    expected = expected_layer[:, None, None] * mixing_ratio
    np.testing.assert_allclose(np.asarray(dtau), expected)


def test_layer_optical_depth_ckd_matches_common_number_column_api():
    dpressure = jnp.array([0.1, 0.2])
    xstensor_ckd = jnp.arange(1.0, 25.0).reshape((2, 3, 4)) * 1.0e-20
    mixing_ratio = jnp.array([1.0e-3, 2.0e-3])
    mass = jnp.array([2.3, 2.4])
    gravity = jnp.array([900.0, 1000.0])

    pressure_result = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )
    absorber_number_column = (
        opacity_factor * dpressure * mixing_ratio / (mass * gravity)
    )
    column_result = layer_optical_depth_from_cross_section(
        xstensor_ckd, absorber_number_column
    )

    np.testing.assert_array_equal(pressure_result, column_result)


def test_layer_optical_depth_ckd_preserves_leading_singleton_broadcasting():
    dpressure = jnp.array([0.1, 0.2])
    xstensor_ckd = jnp.ones((1, 3, 4)) * 1.0e-20
    mixing_ratio = jnp.array([1.0e-3])
    mass = jnp.array([2.3])
    gravity = jnp.array([900.0])

    optical_depth = layer_optical_depth_ckd(
        dpressure, xstensor_ckd, mixing_ratio, mass, gravity
    )
    expected_column = (
        opacity_factor * dpressure * mixing_ratio[0] / (mass[0] * gravity[0])
    )

    assert optical_depth.shape == (2, 3, 4)
    np.testing.assert_allclose(
        optical_depth, xstensor_ckd * expected_column[:, None, None]
    )
