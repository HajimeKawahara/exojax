#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Custom VJP version of the line profile functions used in exospectral
analysis."""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.special.faddeeva import rewofz, imwofz, rewofzx
from exojax.special.faddeeva import wofzs2, rewofzs2, imwofzs2

from jax import custom_vjp


@jit
def ljert(x, a):
    """ljert function, consisting of a combination of imwofz and imwofzs2.

    Args:
        x:
        a:

    Returns:
        ljert: L(x,a) or Imag(wofz(x+ia))

    Examples:

       ljert provides a L(x,a) function.


       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.
    """
    r2 = x*x+a*a
    return jnp.where(r2 < 111., imwofz(x, a), imwofzs2(x, a))


@custom_vjp
def rhjert(x, a):
    """custom VJP version of the Voigt-Hjerting function, consisting of a
    combination of rewofz and real(wofzs2).

    Args:
        x: 
        a:

    Returns:
        rhjert: H(x,a) or Real(wofz(x+ia))

    Examples:

       rhjert provides a Voigt-Hjerting function w/ custom JVP. 

       >>> rhjert(1.0,1.0)
          DeviceArray(0.30474418, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(rhjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(rhjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)
    """
    r2 = x*x+a*a
    return jnp.where(r2 < 111., rewofz(x, a), rewofzs2(x, a))


def h_fwd(x, a):
    dHdx = 2.0*a*ljert(x, a)-2.0*x*rhjert(x, a)
    dHda = 2.0*x*ljert(x, a)+2.0*a*rhjert(x, a)-2.0/jnp.sqrt(jnp.pi)
    res = (dHdx, dHda)
    return rhjert(x, a), res


def h_bwd(res, u):
    dHdx, dHda = res
    return (dHdx * u, dHda * u)


rhjert.defvjp(h_fwd, h_bwd)


@jit
def rvoigtone(nu, sigmaD, gammaL):
    """Custom VJP version of (non-vmapped) Voigt function using Voigt-Hjerting
    function.

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt funtion
    """

    sfac = 1.0/(jnp.sqrt(2)*sigmaD)
    v = sfac*rhjert(sfac*nu, sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v


@jit
def rvoigt(nu, sigmaD, gammaL):
    """Custom VJP version of Voigt profile using Voigt-Hjerting function.

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile
    """

    sfac = 1.0/(jnp.sqrt(2)*sigmaD)
    vhjert = vmap(rhjert, (0, None), 0)
    v = sfac*vhjert(sfac*nu, sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v


@jit
def rvvoigt(numatrix, sigmaD, gammas):
    """Custom VJP version of vmaped voigt profile.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline

    Return:
       Voigt profile vector in R^Nwav
    """
    vmap_voigt = vmap(rvoigt, (0, 0, 0), 0)
    return vmap_voigt(numatrix, sigmaD, gammas)


@jit
def rxsvector(numatrix, sigmaD, gammaL, Sij):
    """Custom VJP version of cross section vector.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline
       Sij: line strength vector in R^Nline

    Return:
       cross section vector in R^Nwav
    """
    return jnp.dot((rvvoigt(numatrix, sigmaD, gammaL)).T, Sij)


@jit
def rxsmatrix(numatrix, sigmaDM, gammaLM, SijM):
    """Custom VJP version of cross section matrix.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    return vmap(rxsvector, (None, 0, 0, 0))(numatrix, sigmaDM, gammaLM, SijM)
