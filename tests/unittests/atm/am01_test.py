from exojax.atm import viscosity
from exojax.atm import atmprof
from exojax.atm import vterm
import jax.numpy as jnp
import pytest
from exojax.atm.amclouds import sigmag_from_effective_radius
from exojax.atm.amclouds import effective_radius
from exojax.atm.amclouds import get_rg

def test_viscosity():
    T = 1000.0  # K
    assert viscosity.eta_Rosner_H2(T) == pytest.approx(0.0001929772857173383)


def test_pressure_scale_height_for_Earth():
    g = 980.0  # cm^2/s
    T = 300.0  # K
    mu = 28.8
    ref = 883764.8664527453

    assert atmprof.pressure_scale_height(g, T, mu) == pytest.approx(ref)


def test_terminal_velocity():
    g = 980.0
    drho = 1.0
    rho = 1.29 * 1.0e-3  # g/cm3
    vfactor, Tr = viscosity.calc_vfactor(atm="Air")
    eta = viscosity.eta_Rosner(300.0, vfactor)
    r = jnp.logspace(-5, 0, 70)
    vfall = vterm.terminal_velocity(r, g, eta, drho, rho)
    assert jnp.mean(vfall) == pytest.approx(328.12296)


def _am01_test_param_set():
    rw = 1.0e-4
    fsed = 2.0
    alpha = 2.0
    sigmag = 2.0

    rg_ref = 2.0695821e-05  # computed from get_rg
    reff_ref = 6.879041e-05  # computed from effective_radius
    return rw, fsed, alpha, sigmag, rg_ref, reff_ref


def test_get_rg():
    rw, fsed, alpha, sigmag, rg_ref, _ = _am01_test_param_set()
    assert get_rg(rw, fsed, alpha, sigmag) == pytest.approx(rg_ref)


def test_effective_radius():
    _, _, _, sigmag, rg_ref, reff_ref = _am01_test_param_set()
    assert effective_radius(rg_ref, sigmag) == pytest.approx(reff_ref)


def test_sigmag_from_effective_radius():
    rw, fsed, alpha, sigmag_ref, rg, reff = _am01_test_param_set()
    val = sigmag_from_effective_radius(reff, fsed, rw, alpha)
    assert val == pytest.approx(sigmag_ref)


if __name__ == "__main__":
    test_sigmag_from_effective_radius()
