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
from exojax.spec import lpf

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
    tau=jnp.dot(lpf.VoigtTc(numatrix,sigmaD,gammaL).T,S)
    f=jnp.exp(-A*tau)
    return f


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

    g = lambda hatnu: lpf.VoigtTc(nu - hatnu,sigmaD,gammaL)
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
    numatrix=lpf.make_numatrix(nuarr,hatnufix,nu0fix)
    ts=time.time()
    ans1=MultiAbsVTc(numatrix,sDfix,gLfix,Afix,Sfix).block_until_ready()
    #    VTcmap=vmap(VoigtTc,(0,None,None),0)
    #    VTcmap(numatrix,sDfix,gLfix)
    te=time.time()
    print(te-ts)
    import matplotlib.pyplot as plt
    plt.plot(ans1)
    plt.show()
    
