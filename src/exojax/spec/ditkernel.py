"""Kernels for Discrete Integral Transform

   * Fourier kernels for the Voigt are given in this module
   * For coarsed wavenumber grids, folded one is needed to avoid negative values, See discussion by Dirk van den Bekerom at https://github.com/radis/radis/issues/186#issuecomment-764465580 for details.

"""

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
#from functools import partial

def voigt_kernel(k, beta,gammaL):
    """Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugated of wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentzian Half Width
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """
    val=(jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]
    return jnp.exp(-2.0*val)

@jit
def f1_voigt_kernel(k,beta,gammaL,dnu):
    """Folded Fourier Kernel of the Voigt Profile for Nfold=1 (not using scan)
  
  
    Args:
        k: conjugate wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        dnu: linear waveunmber grid size
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        This function is the folded voigt kernel but Nfold=1 without lax.scan
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """
    
    dL=1.0/dnu
    val=jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]))
    val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]+dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(k[:,None,None]+dL)))
    val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]-dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(dL-k[:,None,None])))   
    
    return val

@jit
def folded_voigt_kernel(k,beta,gammaL,dLarray):
    """Folded Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugate wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        Nfold: Folding number
        dnu: linear waveunmber grid size
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """

    def ffold(val,dL):
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]+dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(k[:,None,None]+dL)))
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]-dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(dL-k[:,None,None])))
        null=0.0
        return val, null
    
    val=jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]))
    val,nullstack=scan(ffold, val, dLarray)
    
    return val
