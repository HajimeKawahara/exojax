#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Evaluation of molecular lines

"""

from jax import jit, vmap
import jax.numpy as jnp
from jax import custom_jvp
from exojax.special.erfcx import erfcx
import numpy as np

def contfunc(dtau,nu,Parr,dParr,Tarr):
    """contribution function

    Args:
       dtau: delta tau array [N_layer, N_lines]
       nu: wavenumber array [N_lines]
       Parr: pressure array  [N_layer]
       dParr: delta pressure array  [N_layer] 
       Tarr: temperature array  [N_layer]

    """
    tau=np.cumsum(dtau,axis=0)
    hcperk=1.4387773538277202
    cf=np.exp(-tau)*dtau\
        *(dtau*Parr[:,None]/dParr[:,None])\
        *nu**3/(np.exp(hcperk*nu/Tarr[:,None])-1.0)
    return cf


@jit
def voigt0(sigmaD,gammaL):
    """Voigt-Hjerting function at nu=nu0

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile 
       gammaL: broadening coefficient in Lorentz profile 
 
    Returns:
       v: Voigt profile at nu=nu0

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    v=sfac*erfcx(sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v


@jit
def xsvector0(sigmaD,gammaL,Sij):
    """cross section at nu=nu0

    Args:
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline
       Sij: line strength vector in R^Nline

    Return:
       cross section vector in R^Nwav

    """
    vmap_voigt0=vmap(voigt0,(0,0),0)
    return Sij*vmap_voigt0(sigmaD,gammaL)

@jit
def xsmatrix0(sigmaDM,gammaLM,SijM):
    """cross section matrix at nu=nu0

    Args:
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)

    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
    return vmap(xsvector0,(0,0,0))(sigmaDM,gammaLM,SijM)

