"""Ackerman and Marley 2001 cloud model

   - Ackerman and Marley (2001) ApJ 556, 872


"""
from jax import jit
import jax.numpy as jnp
from jax import vmap

def VMRcloud(P,Pbase,fsed,VMRbase,kc=1):
    """VMR of clouds based on AM01
    
    Args:
        P: Pressure (bar)
        Pbase: base pressure (bar)
        fsed: fsed
        VMRbase: VMR of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio
        
    Returns:
        VMR of condensates
        
    """
    VMRc=jnp.where(Pbase>P,VMRbase*(P/Pbase)**(fsed/kc),0.0)
    return VMRc

@jit
def get_VMRc(P,Pbase,fsed,VMRbase,kc=1):
    """VMR array of clouds based on AM01
    
    Args:
        Parr: Pressure array [Nlayer] (bar)
        Pbase: base pressure (bar)
        fsed: fsed
        VMRbase: VMR of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio
        
    Returns:
        VMR array of condensates [Nlayer]
        
    """
    return vmap(VMRcloud,(0,None,None,None),0)(P,Pbase,fsed,VMRbase,kc)


@jit
def get_Pbase(Parr,Psat,VMR):
    """get Pbase from an intersection of a T-P profile and Psat(T) curves
    Args:
        Parr: pressure array
        Psat: saturation pressure arrau
        VMR: VMR for vapor
        
    Returns:
        Pbase: base pressure
    """
    #ibase=jnp.searchsorted((Psat/VMR)-Parr,0.0) # 231 +- 9.2 us
    ibase=jnp.argmin(jnp.abs(jnp.log(Parr)-jnp.log(Psat)+jnp.log(VMR))) # 73.8 +- 2.9 us
    return Parr[ibase]

def vf_stokes(r,g,eta,Nkn=0.):
    """terminal velocity of Stokes flow (Reynolds number << 1)
    
    Args:
        r: particle size
        g: gravity
        eta: dynamic viscosity
        Nkn: Knudsen number
        
    Note:
        (1.0+1.255*Nkn) is the Cunningham factor
    
    Return:
        terminal velocity
        
    """
    return 2.0*g*r*r*(1.0+1.255*Nkn)/(9.0*eta)

def vterminal(drho,eta,Nkn):
    """terminal velocity of droplets

    Note:
       Based on equation (B1) in Appendix B of AM01. See also (10-138) p415 in Hans R Pruppacher and James D Klett. Microstructure of atmospheric clouds and precipitation. InMicrophysics of clouds and precipitation, pages 10–73. Springer, 2010

    Args: 
       drho: density difference between condensates and atmosphere
       eta: dynamic viscocity of the atmosphere
       Nkn: Knudsen number (ratio of the molecular mean free path to the droplet radius)
       
    Returns:
       terminal velocity

    """

    vf=2.0/9.0*(1.0+1.26*Nkn)*drho/eta
    
    return vf
    
