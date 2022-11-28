"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from jax import grad
from exojax.utils.constants import hcperk
from exojax.spec.lbd import weight_point1, weight_point2
from exojax.spec.lbderror import weight_point1_dE, weight_point2_dE
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import single_tilde_line_strength_first
from jax.config import config
config.update("jax_enable_x64", True)

def beta(T, Tref):
    return (hcperk * (1. / T - 1. / Tref))


def f(E, T, Tref):
    return np.exp(-beta(T, Tref) * E)


def error_zeroth_eq_analytic(T, Tref, dE):
    x = -beta(T, Tref) * dE / 2.0
    return 0.5 * (np.exp(x) + np.exp(-x))


def error_zeroth_analytic(T, Ttyp, Tref, dE):
    """analytic formula for the zeroth term of ds (p=0.5).  

    Args:
        T (float): temperature 
        Ttyp (float): typical temperature
        Tref (float): reference temperature
        dE (float): energy interval in cm-1

    Returns:
        float: the zeroth term of ds 
    """
    
    alpha = beta(Ttyp, Tref)
    denom = np.exp(alpha * dE / 2.0) - np.exp(-alpha * dE / 2.0)
    num = (1.0 - np.exp(-alpha * dE / 2.0)) * np.exp(
        beta(T, Tref) * dE / 2.0) - (1.0 - np.exp(alpha * dE / 2.0)) * np.exp(
            -beta(T, Tref) * dE / 2.0)
    return num / denom


def error_first_analytic(T, Ttyp, Tref, dE):
    """analytic formula for the first derivative term of ds (p=0.5).  

    Args:
        T (float): temperature 
        Ttyp (float): typical temperature
        Tref (float): reference temperature
        dE (float): energy interval in cm-1

    Returns:
        float: the first derivative term of ds 
    """
    alpha = beta(Ttyp, Tref)
    facm = np.exp(alpha * dE / 2.0) - np.exp(-alpha * dE / 2.0)
    facp = 2.0 - (np.exp(alpha * dE / 2.0) + np.exp(-alpha * dE / 2.0))
    fac2 = np.exp(beta(T, Tref) * dE / 2.0) - np.exp(-beta(T, Tref) * dE / 2.0)

    return -dE / 2.0 * (alpha - beta(T, Tref)) * fac2 * facp / facm**2


def ds_first(Tarr, Ttyp1, Tref, dE):
    return error_zeroth_analytic(Tarr, Ttyp1, Tref, dE) + error_first_analytic(
        Tarr, Ttyp1, Tref, dE) - 1

def test_single_tilde_line_strength():
    El = 200.0
    dE = 300.0
    p = 1.0 / 2.0
    ttyp = 1.0 / 700.
    tref = 1.0 / 300.0

    for Tt in [250.0, 500.0, 800.0, 1000.0]:
        t = 1.0 / Tt
        ds0 = single_tilde_line_strength_zeroth(t, ttyp, tref, dE, p)
        ds0_ana = error_zeroth_analytic(1. / t, 1. / ttyp, 1. / tref, dE)
        ds1 = single_tilde_line_strength_first(t, ttyp, tref, dE, p)
        ds1_ana = error_first_analytic(1. / t, 1. / ttyp, 1. / tref, dE)
        assert ds0 == pytest.approx(ds0_ana - 1.0)
        assert ds1 == pytest.approx(ds0_ana + ds1_ana - 1.0)


def test_weight_points():
    
    t = 1.0 / 700.0
    tref = 1.0 / 300.0
    El = 200.0
    dE = 300.0
    E1 = El - dE / 3.
    E2 = El + 2.0 * dE / 3.
    w1 = weight_point1(t, tref, El, E1, E2)
    d1 = grad(weight_point1, argnums=0)(t, tref, El, E1, E2)
    w2 = weight_point2(t, tref, El, E1, E2)
    d2 = grad(weight_point2, argnums=0)(t, tref, El, E1, E2)
    assert w1 == pytest.approx(0.75279695)
    assert d1 == pytest.approx(-41.985138)
    assert d1 == -d2


def test_weight_points_dE():
    
    t = 1.0 / 700.0
    tref = 1.0 / 300.0
    El = 200.0
    dE = 300.0
    p = 1.0 / 3.0
    E1 = El - p * dE
    E2 = El + (1.0 - p) * dE
    w1 = weight_point1(t, tref, El, E1, E2)
    d1 = grad(weight_point1, argnums=0)(t, tref, El, E1, E2)
    w2 = weight_point2(t, tref, El, E1, E2)
    d2 = grad(weight_point2, argnums=0)(t, tref, El, E1, E2)
    w1e = weight_point1_dE(t, tref, dE, p)
    d1e = grad(weight_point1_dE, argnums=0)(t, tref, dE, p)
    w2e = weight_point2_dE(t, tref, dE, p)
    d2e = grad(weight_point2_dE, argnums=0)(t, tref, dE, p)
    assert w1e == pytest.approx(w1)
    assert w2e == pytest.approx(w2)
    assert d1e == pytest.approx(d1)
    assert d2e == pytest.approx(d2)


if __name__ == "__main__":
    test_weight_points()
    test_weight_points_dE()
    test_single_tilde_line_strength()
    #test_single_tilde_line_strength_first()