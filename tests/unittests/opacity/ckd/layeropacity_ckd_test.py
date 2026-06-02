import numpy as np

from exojax.rt.layeropacity import layer_optical_depth_ckd
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
