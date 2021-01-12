#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary
--------
line profile functions used in exospectral analysis.

"""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.scipy.special import rewofz# as rewofz
from exojax.scipy.special import rewofzx

@jit
def VoigtRewofz(nu,sigmaD,gammaL):
    """Voigt-Rewofz C profile = Voigt profile using Rewofz function VRewofz(nu,sigmaD,gammaL)

    Args:
       nu: ndarray
            wavenumber
       sigmaD: float
                sigma parameter in Doppler profile 
       gammaL: float 
                broadening coefficient in Lorentz profile 
 
    Returns:
       v: ndarray
           VRewofz

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    vrewofz=vmap(rewofz,(0,None),0)
    v=sfac*vrewofz(sfac*nu,sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v

@jit
def FAbsVRewofz(nu,sigmaD,gammaL,A):
    """
    Summary
    ---------
    Absorption profile using Rewofz (FAbsVRewofz)
    f = exp(-tau)
    tau = A*VRewofz(nu,sigmaD,gammaL)

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
        FAbsVRewofz

    """
    tau=A*VoigtRewofz(nu,sigmaD,gammaL)
    f=jnp.exp(-tau)
    return f


@jit
def VoigtRewofzx(nu,sigmaD,gammaL):
    """[custom] Voigt-Rewofz C profile = Voigt profile using Rewofz function VRewofz(nu,sigmaD,gammaL)

    Args:
       nu: ndarray
            wavenumber
       sigmaD: float
                sigma parameter in Doppler profile 
       gammaL: float 
                broadening coefficient in Lorentz profile 
 
    Returns:
       v: ndarray
           VRewofz

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    vrewofz=vmap(rewofzx,(0,None),0)
    v=sfac*vrewofz(sfac*nu,sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v

@jit
def FAbsVRewofzx(nu,sigmaD,gammaL,A):
    """[custom] 
    Summary
    ---------
    Absorption profile using Rewofz (FAbsVRewofz)
    f = exp(-tau)
    tau = A*VRewofz(nu,sigmaD,gammaL)

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
        FAbsVRewofz

    """
    tau=A*VoigtRewofzx(nu,sigmaD,gammaL)
    f=jnp.exp(-tau)
    return f


@jit
def Tc(a,x,crit=0.1):
    """Tc Function = Tepper-Garc'ia  Function w/ the inner correction: Tc(a,x)
    
    Args:
        a: float/nd array
            parameter 
        x: float/nd array
           parameter

    Returns:
        g: float/ndarary
           Tc(a,x)

    """
    h=jnp.exp(-x*x)
    gg=h - a*(h*h*(4*x**4+7*x**2+4+1.5*x**-2)-1.5*x**-2-1)/x**2/jnp.sqrt(jnp.pi)
    g=jnp.where(jnp.abs(x)<crit,h - 2.0*a/jnp.sqrt(jnp.pi)*(1-2*x**2),gg)
    return g

@jit
def VoigtTc(nu,sigmaD,gammaL):
    """Voigt-Tepper C profile = Voigt profile using Tc function Vtc(nu,sigmaD,gammaL)

    Args:
       nu: ndarray
            wavenumber
       sigmaD: float
                sigma parameter in Doppler profile 
       gammaL: float 
                broadening coefficient in Lorentz profile 
 
    Returns:
       v: ndarray
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
    """Generate numatrix

    Args:
       nu: jnp array
           wavenumber matrix (Nnu,)
       hatnu: jnp array
              line center wavenumber vector (Nline,), where Nm is the number of lines
       nu0: float
            nu0

    Returns:
       f: jnp array or ndarray
          numatrix (Nline,Nnu)

    """

    numatrix=nu[None,:]-hatnu[:,None]-nu0
    return numatrix

