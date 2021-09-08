from exojax.atm import viscosity
from exojax.atm import atmprof
from exojax.atm import vterm
import jax.numpy as jnp

def test_viscosity():
    T=1000.0 #K
    assert viscosity.eta_Rosner_H2(T)==0.0001929772857173383
    
def test_scale_height():
    g=980.0
    T=300.0
    mu=28.8
    assert atmprof.Hatm(g,T,mu)==883764.8664527453

def test_vterm():
    g=980.
    drho=1.0
    rho=1.29*1.e-3 #g/cm3
    vfactor,Tr=viscosity.calc_vfactor(atm="Air")
    eta=viscosity.eta_Rosner(300.0,vfactor)
    r=jnp.logspace(-5,0,70)
    vfall=vterm.vf(r,g,eta,drho,rho)
    assert jnp.mean(vfall)-328.12296 < 1.e-5
