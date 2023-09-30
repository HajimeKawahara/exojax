import pytest
import jax.numpy as jnp
import numpy as np
from exojax.atm.atmprof import normalized_layer_height
from exojax.spec.opachord import chord_geometric_matrix
from exojax.spec.opachord import chord_optical_depth
from exojax.spec.rtransfer import rtrun_trans_pureabs_trapezoid
from jax.config import config

config.update("jax_enable_x64", True)


def test_transmission_pure_absorption_equals_to_Rp_sqaured_for_opaque():
    Nlayer = 5
    Nnu = 2
    dtau_chord = jnp.ones((Nlayer, Nnu)) * jnp.inf
    radius = jnp.array([1.4, 1.3, 1.2, 1.1, 1.0])

    Rp2 = rtrun_trans_pureabs_trapezoid(dtau_chord, radius)

    assert np.all(Rp2 == radius[0]**2 * np.ones(Nnu))

# this test code requires gpu 
def test_chord_geometric_matrix():
    Nlayer = 5
    height = 0.1 * jnp.ones(Nlayer)
    radius = jnp.array([1.4, 1.3, 1.2, 1.1, 1.0])  #radius[-1] = radius_btm

    cgm = chord_geometric_matrix(height, radius)
    print(jnp.sum(cgm))
    #assert jnp.sum(cgm) == pytest.approx(86.49373)


def test_check_parallel_Ax_tauchord():
    A = jnp.array([[7, 0, 0], [4, 5, 0], [1, 2, 3]])
    x = jnp.array([[1, 2, 3], [4, 5, 6]]).T
    n = []
    for k in range(2):
        n.append(jnp.dot(A, x[:, k]))
    n = jnp.array(n).T

    m = chord_optical_depth(A, x)

    assert np.all(m == n)


def test_first_layer_height_from_compute_normalized_radius_profile():
    from exojax.atm.atmprof import pressure_layer_logspace
    pressure, dParr, pressure_decrease_rate = pressure_layer_logspace(
        log_pressure_top=-8., log_pressure_btm=2., nlayer=20)
    T0 = 300.0
    mmw0 = 28.8
    temperature = T0 * np.ones_like(pressure)
    mmw = mmw0 * np.ones_like(pressure)
    radius_btm = 6500.0 * 1.e5
    gravity_btm = 980.

    normalized_height, normalized_radius_lower = normalized_layer_height(
        temperature, pressure_decrease_rate, mmw, radius_btm, gravity_btm)

    normalized_radius_top = normalized_radius_lower[0] + normalized_height[0]
    assert normalized_radius_top == pytest.approx(1.033503571206093)
    assert jnp.sum(normalized_height[1:]) + 1.0 == pytest.approx(
        normalized_radius_lower[0])
    assert normalized_radius_lower[-1] == 1.0


if __name__ == "__main__":
    test_check_parallel_Ax_tauchord()
    test_first_layer_height_from_compute_normalized_radius_profile()
    test_chord_geometric_matrix()
    test_transmission_pure_absorption_equals_to_Rp_sqaured_for_opaque()
