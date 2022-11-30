"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from exojax.utils.constants import hcperk
from jax.config import config
from exojax.spec.lbd import lbd_coefficients
from exojax.spec.lbd import weight

config.update("jax_enable_x64", True)


def example_lbd():
    from jax.config import config
    config.update("jax_enable_x64", True)

    elower_lines = np.array([70.0, 130.0])
    elower_grid = np.array([0.0, 100.0, 200.0])
    Tref = 300.0
    Twt = 700.0
    p1 = 0.7
    p2 = 0.3
    dE = 100.0
    c0, c1, i = lbd_coefficients(elower_lines, elower_grid, Tref, Twt)
    return c0, c1, i, Twt, Tref, dE, p1, p2


def test_lbd_coefficients():
    """We check here consistency with lbderror.weight_point2_dE
    """
    from exojax.spec.lbderror import weight_point2_dE
    from jax import grad
    c0, c1, i, Twt, Tref, dE, p1, p2 = example_lbd()
    cref0a = weight_point2_dE(1.0 / Twt, 1.0 / Tref, dE, p1)
    cref0b = weight_point2_dE(1.0 / Twt, 1.0 / Tref, dE, p2)
    cref1a = grad(weight_point2_dE, argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p1)
    cref1b = grad(weight_point2_dE, argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p2)

    assert c0[0] == pytest.approx(cref0a)
    assert c0[1] == pytest.approx(cref0b)
    assert c1[0] == pytest.approx(cref1a)
    assert c1[1] == pytest.approx(cref1b)


def test_weight():
    c0, c1, i, Twt, Tref, dE, p1, p2 = example_lbd()
    T = 1000.0
    w1, w2 = weight(T, Tref, c0, c1)
    assert np.all(w1 == pytest.approx([0.36565673, 0.76204178]))
    assert np.all(w2 == pytest.approx([0.63434327, 0.23795822]))




def test_construct_lbd():
    return


if __name__ == "__main__":
    test_lbd_coefficients()
    test_weight()