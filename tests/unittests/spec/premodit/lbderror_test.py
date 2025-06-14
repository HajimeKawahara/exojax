"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from jax import grad
from exojax.utils.constants import hcperk
from exojax.opacity.premodit.lbderror import weight_point1_dE, weight_point2_dE
from exojax.opacity.premodit.lbderror import single_tilde_line_strength_zeroth
from exojax.opacity.premodit.lbderror import single_tilde_line_strength_first
from exojax.opacity.premodit.lbderror import worst_tilde_line_strength_first
from exojax.opacity.premodit.lbderror import optimal_params

from jax import config

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


def test_weight_points_dE():

    t = 1.0 / 700.0
    tref = 1.0 / 300.0
    El = 200.0
    dE = 300.0
    p = 1.0 / 3.0
    E1 = El - p * dE
    E2 = El + (1.0 - p) * dE
    w1e = weight_point1_dE(t, tref, dE, p)
    d1e = grad(weight_point1_dE, argnums=0)(t, tref, dE, p)
    w2e = weight_point2_dE(t, tref, dE, p)
    d2e = grad(weight_point2_dE, argnums=0)(t, tref, dE, p)
    w1, w2, d1, d2 = [
        0.752796919848997, 0.247203080151003, -41.9851418044382,
        41.9851418044382
    ]
    assert w1e == pytest.approx(w1)
    assert w2e == pytest.approx(w2)
    assert d1e == pytest.approx(d1)
    assert d2e == pytest.approx(d2)

def test_worst_tilde_line_strength_first():
    Tref = 500.
    Twp = 1000.
    dE = 1000.
    N=10
    Tarr = np.logspace(np.log10(450.), np.log10(1500.), N)
    x = worst_tilde_line_strength_first(Tarr, Twp, Tref, dE)
    assert np.max(x) < 1.e-2

def test_optimal_params():
    Tl_in = 500.0  #K
    Tu_in = 1200.0  #K
    diffmode = 2
    dE, Tl, Tu = optimal_params(Tl_in, Tu_in, diffmode)    
    assert dE == pytest.approx(2475.0)
    assert Tl == pytest.approx(1108.1485374361412) 
    assert Tu == pytest.approx(570.9650338563875)



if __name__ == "__main__":
    #test_weight_points_dE()
    #test_single_tilde_line_strength()
    #test_single_tilde_line_strength_first()
    #test_worst_tilde_line_strength_first()
    test_optimal_params()