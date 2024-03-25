"""Line profile computation using Discrete Integral Transform for free
wavenumber.

* Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekeroma and E.Pannier.
* This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
* The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekeroma.

Note:
   This module is an altanative version of dit.py but potentially have a line center shift by truncation errors. Be careful. See #106.
"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import folded_voigt_kernel
from jax.numpy import index_exp as joi
from exojax.utils.indexing import getix


@jit
def inc3D(w, x, y, z, xv, yv, zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 3D (memory reduced sum).

    Args:
        w: weight (N)
        x: x values (N)
        y: y values (N)
        z: z values (N)
        xv: x grid
        yv: y grid
        zv: z grid            

    Returns:
        lineshape distribution matrix (integrated neighbouring contribution for 3D)

    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n \otimes fz_n, 
        where w_n is the weight, fx_n, fy_n, and fz_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum

    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> zv=jnp.linspace(0,1,17) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> z=np.random.rand(N)
        >>> val=inc3D(w,x,y,z,xv,yv,zv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc3D(x,y,z,xv,yv,zv)*w,axis=3)
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.520862e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 8.418057e-06 %
    """

    cx, ix = getix(x, xv)
    cy, iy = getix(y, yv)
    cz, iz = getix(z, zv)

    a = jnp.zeros((len(xv), len(yv), len(zv)))
    a = a.at[joi[ix, iy, iz]].add(w*(1-cx)*(1-cy)*(1-cz))
    a = a.at[joi[ix, iy+1, iz]].add(w*(1-cx)*cy*(1-cz))
    a = a.at[joi[ix+1, iy, iz]].add(w*cx*(1-cy)*(1-cz))
    a = a.at[joi[ix+1, iy+1, iz]].add(w*cx*cy*(1-cz))
    a = a.at[joi[ix, iy, iz+1]].add(w*(1-cx)*(1-cy)*cz)
    a = a.at[joi[ix, iy+1, iz+1]].add(w*(1-cx)*cy*cz)
    a = a.at[joi[ix+1, iy, iz+1]].add(w*cx*(1-cy)*cz)
    a = a.at[joi[ix+1, iy+1, iz+1]].add(w*cx*cy*cz)

    return a


@jit
def xsvector(nu_lines, sigmaD, gammaL, S, nu_grid, sigmaD_grid, gammaL_grid, dLarray):
    """Cross section vector (DIT/3D version, default) for nu as a free
    parameter.

    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       nu_lines: line center (Nlines)
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    Returns:
       Cross section in the linear nu grid

    Note:
       This version uses inc3D. So, nu grid is also auto-differentiable. However, be careful for the precision of wavenumber grid, because inc3D uses float32 in JAX/GPU. For instance, consider to use dfnus=nus-np.median(nus) and dfnu_lines=mdbCO.nu_lines-np.median(nus) instead of nus (nu_grid) and nu_lines, to mitigate the problem. 

    Example:
       >>> dfnus=nus-np.median(nus)
       >>> dfnu_lines=nu_lines-np.median(nus)
       >>> dnus=nus[1]-nus[0]
       >>> Nfold=3
       >>> dLarray=jnp.linspace(1,Nfold,Nfold)/dnus 
       >>> xs=xsvector3D(nu_lines,sigmaD,gammaL,Sij,nus,sigmaD_grid,gammaL_grid,dLarray)
    """
    Ng_nu = len(nu_grid)
    Ng_sigmaD = len(sigmaD_grid)
    Ng_gammaL = len(gammaL_grid)

    log_sigmaD = jnp.log(sigmaD)
    log_gammaL = jnp.log(gammaL)

    log_sigmaD_grid = jnp.log(sigmaD_grid)
    log_gammaL_grid = jnp.log(gammaL_grid)
    dnu = (nu_grid[-1]-nu_grid[0])/(Ng_nu-1)
    k = jnp.fft.rfftfreq(2*Ng_nu, dnu)
    val = inc3D(S, nu_lines, log_sigmaD, log_gammaL,
                nu_grid, log_sigmaD_grid, log_gammaL_grid)

    valbuf = jnp.vstack([val, jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf, axis=0)
    #vk=voigt_kernel(k, sigmaD_grid,gammaL_grid)
    #vk=f1_voigt_kernel(k, sigmaD_grid,gammaL_grid, dnu)
    vk = folded_voigt_kernel(k, sigmaD_grid, gammaL_grid, dLarray)

    fftvalsum = jnp.sum(fftval*vk, axis=(1, 2))
    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs


@jit
def xsmatrix(nu_lines, sigmaDM, gammaLM, SijM, nu_grid, dgm_sigmaD, dgm_gammaL, dLarray):
    """Cross section matrix for xsvector (DIT/reduced memory version) for nu as
    a free parameter.

    Args:
       nu_lines: line center (Nlines)
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_sigmaD: DIT Grid Matrix for sigmaD R^(Nlayer, NDITgrid)
       dgm_gammaL: DIT Grid Matrix for gammaL R^(Nlayer, NDITgrid)
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    NDITgrid = jnp.shape(dgm_sigmaD)[1]
    Nline = len(nu_lines)
    Mat = jnp.hstack([sigmaDM, gammaLM, SijM, dgm_sigmaD, dgm_gammaL])

    def fxs(x, arr):
        carry = 0.0
        sigmaD = arr[0:Nline]
        gammaL = arr[Nline:2*Nline]
        Sij = arr[2*Nline:3*Nline]
        sigmaD_grid = arr[3*Nline:3*Nline+NDITgrid]
        gammaL_grid = arr[3*Nline+NDITgrid:3*Nline+2*NDITgrid]
        arr = xsvector(nu_lines, sigmaD, gammaL, Sij, nu_grid,
                       sigmaD_grid, gammaL_grid, dLarray)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm
