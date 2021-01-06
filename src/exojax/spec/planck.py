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
