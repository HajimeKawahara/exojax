from exojax.atm import viscosity
from exojax.atm import atmprof
from exojax.atm import vterm
import jax.numpy as jnp
import pytest

def test_viscosity():
    T = 1000.0  # K
    assert viscosity.eta_Rosner_H2(T) == pytest.approx(0.0001929772857173383)


def test_pressure_scale_height_for_Earth():
    g = 980.0 #cm^2/s
    T = 300.0 # K
    mu = 28.8
    
    assert atmprof.pressure_scale_height(g, T, mu) == pytest.approx(883764.8664527453)


def test_terminal_velocity():
    g = 980.
    drho = 1.0
    rho = 1.29*1.e-3  # g/cm3
    vfactor, Tr = viscosity.calc_vfactor(atm='Air')
    eta = viscosity.eta_Rosner(300.0, vfactor)
    r = jnp.logspace(-5, 0, 70)
    vfall = vterm.terminal_velocity(r, g, eta, drho, rho)
    assert jnp.mean(vfall)== pytest.approx(328.12296)

if __name__ == "__main__":
    from exojax.utils.astrofunc import gravity_jupiter
    g = gravity_jupiter(1.0,1.0)
    print(g)
    T=500.
    mu=28.00863
    H = atmprof.pressure_scale_height(g, T, mu)
    print(H)
    import numpy as np
    from exojax.utils.constants import RJ
    dq = np.log(10**1) - np.log(10**-9)
    print(np.exp(H*dq/RJ))