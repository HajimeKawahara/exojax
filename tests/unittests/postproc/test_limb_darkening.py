from exojax.postproc.limb_darkening import (
    average_limb_darkening_coefficients,
    ld_kipping,
    quadratic_ld_from_intensity,
)
from exojax.rt.rtransfer import initialize_gaussian_quadrature
import jax.numpy as jnp
import pytest

def test_ld_kipping():
    u1,u2=ld_kipping(0.5,0.5)
    assert u1 == pytest.approx(0.70710677)
    assert u2 == pytest.approx(0.0)


def test_quadratic_ld_from_intensity():
    mus, weights = initialize_gaussian_quadrature(8)
    mus = jnp.asarray(mus)
    weights = jnp.asarray(weights)
    u1 = jnp.array([0.1, 0.3])
    u2 = jnp.array([0.2, -0.1])
    central_intensity = jnp.array([2.0, 3.0])
    q = 1.0 - mus[:, None]
    intensity = central_intensity * (1.0 - u1 * q - u2 * q**2)

    u1_fit, u2_fit = quadratic_ld_from_intensity(mus, intensity, weights)

    assert u1_fit == pytest.approx(u1, rel=1.0e-4, abs=1.0e-6)
    assert u2_fit == pytest.approx(u2, rel=1.0e-4, abs=1.0e-6)


def test_average_limb_darkening_coefficients():
    u1 = jnp.array([0.1, 0.3])
    u2 = jnp.array([0.2, -0.1])
    weights = jnp.array([1.0, 3.0])

    u1_mean, u2_mean = average_limb_darkening_coefficients(u1, u2, weights)

    assert u1_mean == pytest.approx(0.25)
    assert u2_mean == pytest.approx(-0.025)
