import jax.numpy as jnp
import numpy as np

from exojax.rt.chord import (
    chord_geometric_matrix,
    chord_geometric_matrix_lower,
    chord_optical_depth,
)


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
    np.testing.assert_allclose(ref, cgm, rtol=1.0e-12, atol=1.0e-12)


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
    np.testing.assert_allclose(ref, cgm, rtol=1.0e-12, atol=1.0e-12)


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
