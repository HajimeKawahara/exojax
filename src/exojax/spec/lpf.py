#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary
--------
line profile functions used in exospectral analysis.

"""

from jax import jit
import jax.numpy as jnp

@jit
def Tc(a,x,crit=0.1):
    """
    Summary
    ----------
    Tc Function = Tepper Function w/ the inner correction: Tc(a,x)

    Parameters
    ----------
    a : float/nd array
        parameter 
    x : float/nd array
        parameter

    Returns
    -------
    g : float/ndarary
        Tc(a,x)

    """
    h=jnp.exp(-x*x)
    gg=h - a*(h*h*(4*x**4+7*x**2+4+1.5*x**-2)-1.5*x**-2-1)/x**2/jnp.sqrt(jnp.pi)
    g=jnp.where(jnp.abs(x)<crit,h - 2.0*a/jnp.sqrt(jnp.pi)*(1-2*x**2),gg)
    return g

@jit
def VoigtTc(nu,sigmaD,gammaL):
    """
    Summary
    --------
    Voigt-Tepper C profile = Voigt profile using Tc function Vtc(nu,sigmaD,gammaL)

    Parameters
    ----------
    nu : ndarray
         wavenumber
    sigmaD : float
             sigma parameter in Doppler profile 
    gammaL : float 
             broadening coefficient in Lorentz profile 

    Returns
    -------
    v : ndarray
        Vtc

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    v=sfac*Tc(sfac*gammaL,sfac*nu)/jnp.sqrt(jnp.pi)
    return v

@jit
def FAbsVTc(nu,sigmaD,gammaL,A):
    """
    Summary
    ---------
    Absorption profile using Voigt-Tepper C profile (FAbsVTc)
    f = exp(-tau)
    tau = A*VTc(nu,sigmaD,gammaL)

    Parameters
    ----------
    nu : ndarray
         wavenumber
    sigmaD : float
             sigma parameter in Doppler profile 
    gammaL : float 
             broadening coefficient in Lorentz profile 
    A : float 
        amplitude

    Returns
    -------
    f : ndarray
        FAbsVTc

    """
    tau=A*VoigtTc(nu,sigmaD,gammaL)
    f=jnp.exp(-tau)
    return f

@jit
def make_numatrix(nu,hatnu,nu0):
    """
    Summary
    ----------
    Generate numatrix

    Parameters
    ----------
    nu : jnp array
         wavenumber matrix (Nnu,)
    hatnu : jnp array
         line center wavenumber vector (Nline,), where Nm is the number of lines
    nu0 : float
        nu0

    Returns
    -------
    f : jnp array or ndarray
        numatrix (Nline,Nnu)

    """

    numatrix=nu[None,:]-hatnu[:,None]-nu0
    return numatrix

