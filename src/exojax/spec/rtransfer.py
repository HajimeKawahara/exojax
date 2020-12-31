#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""emission profile functions used in exospectral analysis.

"""

from jax import jit
from exojax.spec import lpf
import jax.numpy as jnp


@jit
def cross(numatrix,sigmaD,gammaL,S):
    """cross section
    Params:
       numatrix: jnp array
                 wavenumber matrix
       sigmaD: float
               sigma parameter in Voigt profile
       gammaL: float
               gamma parameter in Voigt profile
       S: jnp array
          line strength array
    
    Returns:
       cs: cross section
    """
    cs = jnp.dot(lpf.VoigtTc(numatrix,sigmaD,gammaL).T,S)
    return cs

@jit
def calc_dtau(dP,cs,X,m,g):
    dtau=(dP.T*cs)*X/(m*g)
    return dtau

@jit
def calc_tau(dtau):
    return jnp.cumsum(dtau,axis=0)

@jit
def RT1():
    """Multi Emission profile using Voigt-Tepper C profile (MultiAbsVTc)
    sigma = sum_l S_l*VTc(nu -hatnu_k,sigmaD,gammaL)
    """
    #Olson and Kunasz
    f=1
    return f


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


def const_p_layer(logPtop=-2.,logPbtm=2.,NP=17):
    """constructing the pressure layer
    
    ::

    Args: 
       logPtop: float
                log10(P[bar]) at the top layer
       logPbtm: float
                log10(P[bar]) at the bottom layer
       NP: int
                the number of the layers

    ::

    Returns: 
         Parr: jnp array
               pressure layer
         k: float
            k-factor, P[i+1] = k*P[i]
    
    """
    dlogP=(logPbtm-logPtop)/(NP-1)
    k=10**-dlogP
    Parr=jnp.logspace(logPtop,logPbtm,NP)
    Parr=Parr[::-1]
    return Parr, k

def tau_layer(nu,T):
    tau=jnp.dot(lpf.VoigtTc(numatrix,sigmaD,gammaL).T,S)
    lpf.VoigtTc(nu,sigmaD,gammaL)
    dtau=lpf.VoigtTc(nu,sigmaD,gammaL).T
    f=jnp.exp(-A*tau)
    return f
