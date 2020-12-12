#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp

"""lpf

line profile functions used in exospectral analysis.

"""

def Tc(a,x,crit=0.1):
    """
    Tc Function = Tepper Function w/ the inner correction: Tc(a,x)

    Parameters
    ----------
    a : a parameter 
    x : x parameter

    Returns
    -------
    g : Tc(a,x)

    """
    h=jnp.exp(-x*x)
    gg=h - a*(h*h*(4*x**4+7*x**2+4+1.5*x**-2)-1.5*x**-2-1)/x**2/jnp.sqrt(jnp.pi)
    g=jnp.where(jnp.abs(x)<crit,h - 2.0*a/jnp.sqrt(jnp.pi)*(1-2*x**2),gg)
    return g

def VoigtTc(nu,sigmaD,gammaL):
    """
    Voigt-Tepper C profile = Voigt profile using Tc function Vtc(nu,sigmaD,gammaL)

    Parameters
    ----------
    nu : wavenumber
    sigmaD : sigma parameter in Doppler profile 
    gammaL : broadening coefficient in Lorentz profile 

    Returns
    -------
    v : Vtc

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    v=sfac*Tc(sfac*gammaL,sfac*nu)/jnp.sqrt(jnp.pi)
    return v

def FAbsVTc(nu,sigmaD,gammaL,A):
    """
    Absorption profile using Voigt-Tepper C profile (FAbsVTc)
    f = exp(-tau)
    tau = A*VTc(nu,sigmaD,gammaL)

    Parameters
    ----------
    nu : wavenumber
    sigmaD : sigma parameter in Doppler profile 
    gammaL : broadening coefficient in Lorentz profile 
    A : amplitude

    Returns
    -------
    f : FAbsVTc

    """
    tau=A*VoigtTc(nu,sigmaD,gammaL)
    f=jnp.exp(-tau)
    return f
