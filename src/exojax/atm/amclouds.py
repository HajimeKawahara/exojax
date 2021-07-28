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


