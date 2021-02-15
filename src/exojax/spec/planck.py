from jax import jit
import jax.numpy as jnp

@jit
def nB(T,numic):
    """normalized Planck Function

    Args:
       T: float
          temperature [K]
       numic: float
              wavenumber normalized by nu at 1 micron

    Returns:
           nB: float 
               normalized planck function
    """
    hparkB_mic=14387.769
    return numic**3/(jnp.exp(hparkB_mic*numic/T)-1)



def piBarr(T,nus):
    """pi B array (Planck Function)

    Args:                                                                       
       T: temperature [K]
       nus: wavenumber [cm-1]

    Returns:                                                                    
           jnp.array: pi B (cgs unit) [Nlayer x Nnu]
           
    """
    hcperk=1.4387773538277202
    fac=3.741771790075259e-05
    return (fac*nus**3)/(jnp.exp(hcperk*nus/T[:,None])-1.0)
