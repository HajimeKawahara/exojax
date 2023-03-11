import jax.numpy as jnp
import numpy as np

from jax.lax import scan
from exojax.atm.atmprof import pressure_scale_height


#atmprof
def compute_normalized_radius_profile(temperature, pressure, dParr, mmw_profile,
                             radius_btm, gravity_btm):
    """compute normalized radius at the upper boundary of the atmospheric layer, neglecting atmospheric mass. 

    Args:
        temperature (1D array): temperature profile (K) of the layer, (Nlayer, from atmospheric top to bottom)
        pressure (1D array): pressure profile (bar) of the layer, (Nlayer, from atmospheric top to bottom)
        dParr (1D array): pressure difference profile (bar) of the layer, (Nlayer, from atmospheric top to bottom)
        mmw_profile (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom) 
        radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
        gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

    Returns:
        1D array (Nlayer) : radius profile normalized by radius_btm
    """

    inverse_Tarr = temperature[::-1]
    inverse_dlogParr = (dParr / pressure)[::-1]
    inverse_mmr_arr = mmw_profile[::-1]
    Mat = jnp.hstack([inverse_Tarr, inverse_dlogParr, inverse_mmr_arr])

    def compute_radius(normalized_radius, arr):
        T_layer = arr[0:1]
        dlogP_layer = arr[1:2]
        mmw_layer = arr[2:3]

        gravity_layer = gravity_btm / normalized_radius
        normalized_radius += pressure_scale_height(
            gravity_layer, T_layer, mmw_layer) * dlogP_layer / radius_btm

        return normalized_radius, normalized_radius

    _, normalized_radius_profile = scan(compute_radius, 1.0, Mat)

    return normalized_radius_profile


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


if __name__ == "__main__":
    #test_check_parallel_Ax_tauchord()
    test_inverse_cumsum()