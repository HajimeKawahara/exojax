"""response

   * input nus/wav should be spaced evenly on a log scale (ESLOG).
   * response is a response operation for the wavenumber grid spaced evenly on a log scale.
   * rigidrot2 and ipgauss2 are faster than default when N >~ 10000, where N is the dimension of the wavenumber grid.
   * response uses jax.numpy.convolve, therefore, convolve in cuDNN. 


"""

import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def rigidrot(nus,F0,vsini,u1=0.0,u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)


    """
    c=299792.458
    dvmat=jnp.array(c*jnp.log(nus[None,:]/nus[:,None]))
    x=dvmat/vsini
    x2=x*x
    kernel=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2+2.0*u2*x2),0.0)
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0

    return F

@jit
def ipgauss(nus,F0,beta):
    """Apply the Gaussian IP response to a spectrum F using jax.lax.scan

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)


    """

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nus[None,:]/nus[:,None]))    
    kernel=jnp.exp(-(dvmat)**2/(2.0*beta**2))
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0
    return F

@jit
def ipgauss_sampling(nusd,nus,F0,beta,RV):
    """Apply the Gaussian IP response + sampling to a spectrum F 

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)

    Return:
        response-applied spectrum (F)


    """

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nusd[None,:]/nus[:,None]))    
    kernel=jnp.exp(-(dvmat+RV)**2/(2.0*beta**2))    
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0
    return F



@jit
def rigidrot2(nus,F0,varr_kernel,vsini,u1=0.0,u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)


    """
    x=varr_kernel/vsini
    x2=x*x
    kernel=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2*2.0*u2*x2),0.0)
    kernel=kernel/jnp.sum(kernel,axis=0)
    F=jnp.convolve(F0,kernel,mode="same")

    return F

@jit
def ipgauss2(nus,F0,varr_kernel,beta):
    """Apply the Gaussian IP response to a spectrum F

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)


    """
    x=varr_kernel/beta
    kernel=jnp.exp(-x*x/2.0)
    kernel=kernel/jnp.sum(kernel,axis=0)
    F=jnp.convolve(F0,kernel,mode="same")

    return F

@jit
def sampling(nusd,nus,F,RV):
    """Sampling w/ RV

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber
        F: input spectrum
        RV: radial velocity (km/s)

    Returns:
       sampled spectrum
    
    """
    c=299792.458
    return jnp.interp(nusd*(1.0+RV/c),nus,F)
    
