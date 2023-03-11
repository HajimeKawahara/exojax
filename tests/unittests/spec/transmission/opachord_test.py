import pytest
import jax.numpy as jnp
import numpy as np
from exojax.atm.atmprof import normalized_layer_height

#normalized_radius = normalized_radius_profile(temperature, pressure, dParr,
#                                              mmw, radius_btm, gravity_btm)


def chord_geometric_matrix(layer_height):
    return


def tauchord(chord_geometric_matrix, xsmatrix):
    """chord opacity vector from a chord geometric matrix and xsmatrix

    Note:
        transposed upper triangle matrix is like this
        [[1 2 3]
        [4 5 0]
        [7 0 0]]
        can be obtained by jnp.triu(jnp.array(square_matrix[:, ::-1]))[:,::-1]

    Args:
        chord_geometric_matrix (jnp array): chord geometric matrix (Nlayer, Nlayer), which is converted to a transposed upper triangle matrix 
        xsmatrix (jnp array): cross section matrix (Nlayer, N_wavenumber)

    Returns: tauchord matrix (Nlayer, N_wavenumber)

    """
    return jnp.dot(
        jnp.triu(chord_geometric_matrix[:, ::-1])[:, ::-1], xsmatrix)


def inverse_cumsum(arr):
    return jnp.cumsum(arr[::-1])[::-1]


def test_inverse_cumsum():
    Tarr = jnp.array([0, 1, 2, 3, 4])

    vec = inverse_cumsum(Tarr)

    ans = jnp.array([10, 10, 9, 7, 4])
    assert jnp.all(vec == ans)


def test_check_parallel_Ax_tauchord():
    A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = jnp.array([[1, 2, 3], [4, 5, 6]]).T
    Atranstriu = jnp.array([[1, 2, 3], [4, 5, 0], [7, 0, 0]])
    n = []
    for k in range(2):
        n.append(jnp.dot(Atranstriu, x[:, k]))
    n = jnp.array(n).T

    m = tauchord(A, x)

    assert np.all(m == n)


def test_first_layer_height_from_compute_normalized_radius_profile():
    from exojax.spec.rtransfer import pressure_layer
    pressure, dParr, k = pressure_layer(log_pressure_top=-8.,
                                        log_pressure_btm=2.,
                                        NP=20)
    T0 = 300.0
    mmw0 = 28.8
    temperature = T0 * np.ones_like(pressure)
    mmw = mmw0 * np.ones_like(pressure)
    radius_btm = 6500.0 * 1.e5
    gravity_btm = 980.

    normalized_height = normalized_layer_height(temperature, pressure, dParr,
                                                mmw, radius_btm, gravity_btm)

    d_scale_height = (normalized_height[-1] + 1.0) * radius_btm
    ref = 650620740.0  #cm
    assert d_scale_height == pytest.approx(ref)


if __name__ == "__main__":
    #test_check_parallel_Ax_tauchord()
    #test_inverse_cumsum()
    test_first_layer_height_from_compute_normalized_radius_profile()