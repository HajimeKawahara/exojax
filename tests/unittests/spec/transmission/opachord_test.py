import pytest
import jax.numpy as jnp
import numpy as np
from exojax.atm.atmprof import normalized_layer_height
from exojax.spec.opachord import chord_geometric_matrix
from exojax.spec.opachord import tauchord


def test_chord_geometric_matrix():
    Nlayer = 5
    height = 0.1 * jnp.ones(Nlayer)
    radius = jnp.cumsum(height)[::-1] + 1.0
    cgm = chord_geometric_matrix(height, radius)
    assert jnp.sum(cgm) == pytest.approx(32.06652 * 2.0)


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
    test_check_parallel_Ax_tauchord()
    test_first_layer_height_from_compute_normalized_radius_profile()
    test_chord_geometric_matrix()