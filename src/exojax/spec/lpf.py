#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary
--------
line profile functions used in exospectral analysis.

"""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.special.faddeeva import rewofz,rewofzs2,rewofzx

@jit
def hjert(x,a):
    """Voigt-Hjerting function, consisting of a combination of rewofz and rewofzs2.
    
    Args:
        x: 
        a:
        
    Returns:
        hjert: H(x,a) or Real(wofz(x+ia))

    Examples:
       
       hjert provides a Voigt-Hjerting function. 
       
       >>> hjert(1.0,1.0)
          DeviceArray(0.3047442, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)

    """
    r2=x*x+a*a
    return jnp.where(r2<111., rewofz(x,a), rewofzs2(x,a))

@jit
def voigt(nu,sigmaD,gammaL):
    """Voigt profile using Voigt-Hjerting function 

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile 
       gammaL: broadening coefficient in Lorentz profile 
 
    Returns:
       v: Voigt profile

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    vrewofz=vmap(hjert,(0,None),0)
    v=sfac*vrewofz(sfac*nu,sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v

@jit
def lorentz(nu,gammaL):
    def f(nu,gammaL):
        return gammaL/(gammaL**2 + nu**2)/jnp.pi
    return vmap(f,(0,None),0)

@jit
def FAbsVHjert(nu,sigmaD,gammaL,A):
    """Absorption profile using Hjert (FAbsVHjert)

    f = exp(-tau)
    tau = A*VRewofz(nu,sigmaD,gammaL)

    Params:
       nu : ndarray
         wavenumber
       sigmaD : float
             sigma parameter in Doppler profile 
       gammaL : float 
             broadening coefficient in Lorentz profile 
       A : float 
             amplitude

    Returns:
       f : ndarray
           FAbsVHjert

    """
    tau=A*voigt(nu,sigmaD,gammaL)
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



