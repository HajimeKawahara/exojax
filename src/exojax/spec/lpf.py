#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary
--------
line profile functions used in exospectral analysis.

"""

from jax import jit, vmap
from jax.lax import map
import jax.numpy as jnp

@jit
def Tc(a,x,crit=0.1):
    """
    Summary
    ---------
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
def MultiAbsVTc(numatrix,sigmaD,gammaL,A,S):
    """
    Summary
    ----------
    Multi Absorption profile using Voigt-Tepper C profile (MultiAbsVTc)
    f = exp(-tau)
    tau = sum_k A*S_k*VTc(nu -hatnu_k,sigmaD,gammaL)

    Parameters
    ----------
    numatrix : jnp array
         wavenumber matrix (Nm, Nnu)
    sigmaD : float
             sigma parameter in Doppler profile 
    gammaL : float 
             broadening coefficient in Lorentz profile 
    A : float
        amplitude
    S : float
        line strength
    hatnu : ndarray
            line center

    Returns
    -------
    f : ndarray
        MultiAbsVTc

    """
    tau=jnp.dot(VoigtTc(numatrix,sigmaD,gammaL).T,S)
    f=jnp.exp(-A*tau)
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
         line center wavenumber vector (Nm,)
    nu0 : float
        nu0

    Returns
    -------
    f : jnp array or ndarray
        numatrix

    """

    numatrix=nu[None,:]-hatnu[:,None]-nu0
    return numatrix


@jit
def MultiAbsVTc_Each(nu,sigmaD,gammaL,A,S,hatnu):
    """
    Summary
    ----------
    Slow version Multi Absorption profile using Voigt-Tepper C profile (MultiAbsVTc)
    f = exp(-tau)
    tau = sum_k A*S_k*VTc(nu -hatnu_k,sigmaD,gammaL)

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
    S : float
        line strength
    hatnu : ndarray
            line center

    Returns
    -------
    f : ndarray
        MultiAbsVTc

    Usage
    ---------
    ```
    f = lambda nu: MultiAbsVTc(nu-nu0fix,sDfix,gLfix,Afix,Sfix,hatnufix)
    ans0=map(f,nuarr).block_until_ready()
    ```

    """

    g = lambda hatnu: VoigtTc(nu - hatnu,sigmaD,gammaL)
    sigmaeach=map(g,hatnu)
    tau=jnp.dot(S,sigmaeach)
    f=jnp.exp(-A*tau)
    return f


if __name__=="__main__":
    import numpy as np
    import time
    np.random.seed(38)
    N=1000
    nur=200
#    nuarr=jnp.linspace(-nur,nur,N)
    nuarr=jnp.linspace(-nur,nur,N)

    sigin=0.01
    sDfix = jnp.array(1.0)
    gLfix = jnp.array(0.5)

    Nmol=1000
    hatnufix = jnp.array((np.random.rand(Nmol)-0.5)*nur*2)
    Sfix=jnp.array(np.random.rand(Nmol))
    print(jnp.shape(Sfix))
    Afix=jnp.array(0.03)
    nu0fix = 0.7
#    numatrix=nuarr[None,:]-hatnufix[:,None]-nu0fix    
    numatrix=make_numatrix(nuarr,hatnufix,nu0fix)
    ts=time.time()
    ans1=MultiAbsVTc(numatrix,sDfix,gLfix,Afix,Sfix).block_until_ready()
    #    VTcmap=vmap(VoigtTc,(0,None,None),0)
    #    VTcmap(numatrix,sDfix,gLfix)
    te=time.time()
    print(te-ts)
    import matplotlib.pyplot as plt
    plt.plot(ans1)
    plt.show()
    
