import jax.numpy as jnp
import pytest

from exojax.atm import viscosity, vterm


def test_terminal_velocity():
    g = 980.0
    rho_cloud = 1.0
    rho = 1.29 * 1.0e-3  # g/cm3
    vfactor, Tr = viscosity.calc_vfactor(atm="Air")
    eta = viscosity.eta_Rosner(300.0, vfactor)
    r = jnp.logspace(-5, 0, 70)
    vfall = vterm.terminal_velocity(r, g, eta, rho_cloud, rho)
    assert jnp.mean(vfall) == pytest.approx(328.12296)
