"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from exojax.utils.constants import hcperk
from jax import config
from exojax.opacity.premodit.lbd import lbd_coefficients
from exojax.opacity.premodit.lbd import weight

config.update("jax_enable_x64", True)


def example_lbd_1():
    from jax import config
    config.update("jax_enable_x64", True)
    Twt = 700.0
    Tref = 300.0
    elower_lines = np.array([70.0, 130.0])
    elower_grid = np.array([0.0, 100.0, 200.0])
    p1 = 0.7
    p2 = 0.3
    dE = 100.0
    coeff, i = lbd_coefficients(elower_lines, elower_grid, Tref, Twt)
    c0, c1, c2 = coeff
    return c0, c1, c2, i, Twt, Tref, dE, p1, p2


def example_lbd_2():
    from jax import config
    config.update("jax_enable_x64", True)
    Twt = 300.0
    Tref = 700.0
    elower_lines = np.array([70.0, 230.0])
    elower_grid = np.array([0.0, 100.0, 200.0, 300.0])
    p1 = 0.7
    p2 = 0.3
    dE = 100.0
    coeff, i = lbd_coefficients(elower_lines, elower_grid, Tref, Twt)
    c0, c1, c2 = coeff
    return c0, c1, c2, i, Twt, Tref, dE, p1, p2


def test_lbd_coefficients():
    """We check here consistency with lbderror.weight_point2_dE
    """
    from exojax.opacity.premodit.lbderror import weight_point2_dE
    from jax import grad
    c0, c1, c2, i, Twt, Tref, dE, p1, p2 = example_lbd_1()
    cref0a = weight_point2_dE(1.0 / Twt, 1.0 / Tref, dE, p1)
    cref0b = weight_point2_dE(1.0 / Twt, 1.0 / Tref, dE, p2)
    cref1a = grad(weight_point2_dE, argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p1)
    cref1b = grad(weight_point2_dE, argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p2)
    cref2a = grad(grad(weight_point2_dE, argnums=0),
                  argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p1)
    cref2b = grad(grad(weight_point2_dE, argnums=0),
                  argnums=0)(1.0 / Twt, 1.0 / Tref, dE, p2)
    assert c0[0] == pytest.approx(cref0a)
    assert c0[1] == pytest.approx(cref0b)
    assert c1[0] == pytest.approx(cref1a)
    assert c1[1] == pytest.approx(cref1b)
    assert c2[0] == pytest.approx(cref2a)
    assert c2[1] == pytest.approx(cref2b)


def test_weight_normal():
    Twt = 700.0
    Tref = 300.0
    c0, c1, c2, i, Twt, Tref, dE, p1, p2 = example_lbd_1()
    T = 1000.0
    w1, w2 = weight(T, Tref, c0, c1)
    assert np.all(w1 == pytest.approx([0.36565673, 0.76204178]))
    assert np.all(w2 == pytest.approx([0.63434327, 0.23795822]))


def test_weight_inverse():
    Twt = 300.0
    Tref = 700.0
    c0, c1, c2, i, Twt, Tref, dE, p1, p2 = example_lbd_2()
    T = 1000.0
    w1, w2 = weight(T, Tref, c0, c1)


def test_construct_lbd():
    return


if __name__ == "__main__":
    #test_weight_inverse()
    test_lbd_coefficients()
    #test_weight_normal()