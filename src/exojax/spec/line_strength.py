import jax.numpy as jnp
from astropy import constants as const
from astropy import units as u

def gf2Sh(gf,nui,hatnu,QT,T):
    """line strength in the unit of cm2/s/species. see Sharps & Burrows equation(1)
    all quantities should be converted to the cgs unit
    Params:
        gf   : g(statistical weight) * f(oscillator strength)
        nui  : initial wavenumber in cm-1
        hatnu: line position in cm-1
        QT: partition function 
        T   : temperature 
    """
    
    eesu=const.e.esu
    ee=(eesu*eesu).to(u.g*u.cm*u.cm*u.cm/u.s/u.s).value
    me=const.m_e.cgs.value
    c=const.c.cgs.value
    h=const.h.cgs.value
    k=const.k_B.cgs.value
    
    fac0=jnp.pi*ee/me/c
    fac1=-h*c*nui/k/T
    fac2=-h*c*hatnu/k/T
    Snorm=fac0*gf*jnp.exp(fac1)/QT*(-jnp.expm1(fac2))

    #jnp.expm1(x) = exp(x) -1 
    
    return Snorm
