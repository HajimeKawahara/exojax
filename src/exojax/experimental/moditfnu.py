"""Line profile computation using Discrete Integral Transform.

* Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekeroma and E.Pannier.
* This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
* The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekeroma.
* See also `DIT for non evenly-spaced linear grid <https://github.com/dcmvdbekerom/discrete-integral-transform/blob/master/demo/discrete_integral_transform_log.py>`_ by  D.C.M van den Bekeroma, as a reference of this code.

Note:
   This module is an altanative version of modit.py but potentially have a line center shift by truncation errors. Be careful. See #106.
"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import folded_voigt_kernel_logst
from jax.numpy import index_exp as joi
from exojax.utils.indexing import getix


@jit
def inc2D(w, x, y, xv, yv):
    """integrated neighbouring contribution function for 2D (memory reduced
    sum).

    Args:
        w: weight (N)
        x: x values (N)
        y: y values (N)
        xv: x grid
        yv: y grid

    Returns:
        integrated neighbouring contribution function

    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n, 
        where w_n is the weight, fx_n and fy_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum

    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> val=inc2D(w,x,y,xv,yv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc2D(x,y,xv,yv)*w,axis=2)        
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.196106e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 1.6135311e-05 %
    """

    cx, ix = getix(x, xv)
    cy, iy = getix(y, yv)
    ncfarray = jnp.zeros((len(xv), len(yv)))
    ncfarray = ncfarray.at[joi[ix, iy]].add(w*(1.-cx)*(1.-cy))
    ncfarray = ncfarray.at[joi[ix, iy+1]].add(w*(1.-cx)*cy)
    ncfarray = ncfarray.at[joi[ix+1, iy]].add(w*cx*(1.-cy))
    ncfarray = ncfarray.at[joi[ix+1, iy+1]].add(w*cx*cy)
    return ncfarray


@jit
def xsvector(nu_lines, nsigmaD, ngammaL, S, nu_grid, ngammaL_grid, dLarray, dv_lines, dv_grid):
    """Cross section vector (DIT/3D version)

    The original code is rundit_fold_logredst in [addit package](https://github.com/HajimeKawahara/addit). DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       nu_lines: line center (Nlines)
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       dv_lines: delta wavenumber for lines i.e. nu_lines/R
       dv_grid: delta wavenumber for nu_grid i.e. nu_grid/R

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
    """

    Ng_nu = len(nu_grid)
    Ng_gammaL = len(ngammaL_grid)

    log_nstbeta = jnp.log(nsigmaD)
    log_ngammaL = jnp.log(ngammaL)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu, 1)
    Slsd = inc2D(S, nu_lines, log_ngammaL, nu_grid,
                 log_ngammaL_grid)  # Lineshape Density
    Sbuf = jnp.vstack([Slsd, jnp.zeros_like(Slsd)])
    til_Slsd = jnp.fft.rfft(Sbuf, axis=0)

    til_Voigt = folded_voigt_kernel_logst(
        k, log_nstbeta, log_ngammaL_grid, dLarray)
    fftvalsum = jnp.sum(til_Slsd*til_Voigt, axis=(1,))

    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]/dv_grid

    return xs


@jit
def xsmatrix(nu_lines, nsigmaDl, ngammaLM, SijM, nu_grid, dgm_ngammaL, dLarray, dv_lines, dv_grid):
    """Cross section matrix for xsvector (MODIT)

    Args:
       nu_lines: line center (Nlines)
       nsigmaDl: normalized doppler sigma in layers in R^(Nlayer x 1)
       ngammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_ngammaL: DIT Grid Matrix for normalized gammaL R^(Nlayer, NDITgrid)
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       dv_lines: delta wavenumber for lines i.e. nu_lines/R
       dv_grid: delta wavenumber for nu_grid i.e. nu_grid/R

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    NDITgrid = jnp.shape(dgm_ngammaL)[1]
    Nline = len(nu_lines)
    Mat = jnp.hstack([nsigmaDl, ngammaLM, SijM, dgm_ngammaL])

    def fxs(x, arr):
        carry = 0.0
        nsigmaD = arr[0:1]
        ngammaL = arr[1:Nline+1]
        Sij = arr[Nline+1:2*Nline+1]
        ngammaL_grid = arr[2*Nline+1:2*Nline+NDITgrid+1]
        arr = xsvector(nu_lines, nsigmaD, ngammaL, Sij, nu_grid,
                       ngammaL_grid, dLarray, dv_lines, dv_grid)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm
