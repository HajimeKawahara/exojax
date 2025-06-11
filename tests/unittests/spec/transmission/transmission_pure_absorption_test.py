import pytest
import jax.numpy as jnp
import numpy as np
from exojax.atm.atmprof import normalized_layer_height
from exojax.rt.chord import chord_geometric_matrix
from exojax.rt.chord import chord_geometric_matrix_lower
from exojax.rt.chord import chord_optical_depth

from exojax.rt.rtransfer import rtrun_trans_pureabs_trapezoid
from jax import config

config.update("jax_enable_x64", True)


def test_transmission_pure_absorption_equals_to_Rp_sqaured_for_opaque():
    Nlayer = 5
    Nnu = 2
    dtau_chord = jnp.ones((Nlayer, Nnu)) * jnp.inf
    radius_lower = jnp.array([1.4, 1.3, 1.2, 1.1, 1.0])
    radius_top = 1.5
    Rp2 = rtrun_trans_pureabs_trapezoid(dtau_chord, radius_lower, radius_top)
    assert Rp2[0] > radius_lower[0]**2 
    assert Rp2[0] < radius_top**2
    
# this test code requires gpu 
def test_chord_geometric_matrix_lower():
    Nlayer = 3
    height = jnp.array([0.15, 0.1, 0.1])
    radius_lower = jnp.array([1.2, 1.1, 1.0])  
    radius_upper = radius_lower + height
    cgm = chord_geometric_matrix_lower(height, radius_lower)
    ref = np.zeros((Nlayer,Nlayer))
    ref[0,0]=2*jnp.sqrt(radius_upper[0]**2 - radius_lower[0]**2)/height[0]
    ref[1,0]=_manual_coeff(radius_upper, radius_lower, radius_lower, height, 1, 0)
    ref[1,1]=2*jnp.sqrt(radius_upper[1]**2 - radius_lower[1]**2)/height[1]
    ref[2,0]=_manual_coeff(radius_upper, radius_lower, radius_lower, height, 2, 0)
    ref[2,1]=_manual_coeff(radius_upper, radius_lower, radius_lower, height, 2, 1)
    ref[2,2]=2*jnp.sqrt(radius_upper[2]**2 - radius_lower[2]**2)/height[2]
    assert np.all(ref == cgm)

def test_chord_geometric_matrix():
    Nlayer = 3
    height = jnp.array([0.15, 0.1, 0.1])
    radius_lower = jnp.array([1.2, 1.1, 1.0])  
    radius_mid = radius_lower + height/2.0
    radius_upper = radius_lower + height
    cgm = chord_geometric_matrix(height, radius_lower)
    ref = np.zeros((Nlayer,Nlayer))
    ref[0,0]=2*jnp.sqrt(radius_upper[0]**2 - radius_mid[0]**2)/height[0]
    ref[1,0]=_manual_coeff(radius_upper, radius_lower, radius_mid, height, 1, 0)
    ref[1,1]=2*jnp.sqrt(radius_upper[1]**2 - radius_mid[1]**2)/height[1]
    ref[2,0]=_manual_coeff(radius_upper, radius_lower, radius_mid, height, 2, 0)
    ref[2,1]=_manual_coeff(radius_upper, radius_lower, radius_mid, height, 2, 1)
    ref[2,2]=2*jnp.sqrt(radius_upper[2]**2 - radius_mid[2]**2)/height[2]
    #assert np.all(ref == cgm)

def _manual_coeff(radius_upper, radius_lower, radius_ref, height, n, k):
    return 2*(jnp.sqrt(radius_upper[k]**2 - radius_ref[n]**2) - jnp.sqrt(radius_lower[k]**2 - radius_ref[n]**2))/height[k]

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
    assert normalized_radius_top == pytest.approx(1.0340775666464417)
    assert jnp.sum(normalized_height[1:]) + 1.0 == pytest.approx(
        normalized_radius_lower[0])
    assert normalized_radius_lower[-1] == 1.0


if __name__ == "__main__":
    #test_check_parallel_Ax_tauchord()
    test_first_layer_height_from_compute_normalized_radius_profile()
    #test_chord_geometric_matrix_lower()
    #test_chord_geometric_matrix()
    #test_transmission_pure_absorption_equals_to_Rp_sqaured_for_opaque()
