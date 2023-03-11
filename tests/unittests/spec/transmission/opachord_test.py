import pytest
import jax.numpy as jnp
import numpy as np
from exojax.atm.atmprof import normalized_layer_height

#normalized_radius = normalized_radius_profile(temperature, pressure, dParr,
#                                              mmw, radius_btm, gravity_btm)


def chord_geometric_matrix(height, radius):
    """compute chord geometric matrix

    Args:
        height (1D array): (normalized) height of the layers from top atmosphere, Nlayer
        radius (1D array): (normalized) radius of the layers from top atmosphere, Nlayer

    Returns:
        2D array: chord geometric matrix (Nlayer, Nlayer)
    """
    radius_roll = jnp.roll(radius, 1)

    # elements at the top layer to be zero
    radius_roll = radius_roll.at[0].set(radius[0])
    height = height.at[0].set(jnp.inf)

    fac_right = jnp.sqrt(radius[None, :]**2 - radius[:, None]**2)
    fac_left = jnp.sqrt(radius_roll[None, :]**2 - radius[:, None]**2)

    raw_matrix = (fac_left - fac_right) / height
    return jnp.tril(raw_matrix)


def test_chord_geometric_matrix():
    Nlayer = 5
    height = 0.1 * jnp.ones(Nlayer)
    radius = jnp.cumsum(height)[::-1] + 1.0
    cgm = chord_geometric_matrix(height, radius)
    print(cgm)


def tauchord(chord_geometric_matrix, xsmatrix):
    """chord opacity vector from a chord geometric matrix and xsmatrix
    
    Args:
        chord_geometric_matrix (jnp array): chord geometric matrix (Nlayer, Nlayer), lower triangle matrix 
        xsmatrix (jnp array): cross section matrix (Nlayer, N_wavenumber)

    Returns: tauchord matrix (Nlayer, N_wavenumber)

    """
    return jnp.dot(chord_geometric_matrix, xsmatrix)


def inverse_cumsum(arr):
    return jnp.cumsum(arr[::-1])[::-1]


def test_inverse_cumsum():
    Tarr = jnp.array([0, 1, 2, 3, 4])

    vec = inverse_cumsum(Tarr)

    ans = jnp.array([10, 10, 9, 7, 4])
    assert jnp.all(vec == ans)


def test_check_parallel_Ax_tauchord():
    A = jnp.array([[7, 0, 0], [4, 5, 0], [1, 2, 3]])
    x = jnp.array([[1, 2, 3], [4, 5, 6]]).T
    n = []
    for k in range(2):
        n.append(jnp.dot(A, x[:, k]))
    n = jnp.array(n).T

    m = tauchord(A, x)

    assert np.all(m == n)


def test_first_layer_height_from_compute_normalized_radius_profile():
    from exojax.atm.atmprof import pressure_layer_logspace
    pressure, dParr, k = pressure_layer_logspace(log_pressure_top=-8.,
                                                 log_pressure_btm=2.,
                                                 NP=20)
    T0 = 300.0
    mmw0 = 28.8
    temperature = T0 * np.ones_like(pressure)
    mmw = mmw0 * np.ones_like(pressure)
    radius_btm = 6500.0 * 1.e5
    gravity_btm = 980.

    normalized_height, normalized_radius = normalized_layer_height(
        temperature, pressure, dParr, mmw, radius_btm, gravity_btm)

    ref = 650620740.0  #cm
    first_height = (normalized_height[-1] + 1.0) * radius_btm
    assert first_height == pytest.approx(ref)
    first_height = normalized_radius[-1] * radius_btm
    assert first_height == pytest.approx(ref)


if __name__ == "__main__":
    #test_check_parallel_Ax_tauchord()
    #test_inverse_cumsum()
    #test_first_layer_height_from_compute_normalized_radius_profile()
    test_chord_geometric_matrix()