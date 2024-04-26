"""Real space evaluation of DIT (REDIT)"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import folded_voigt_kernel_logst
from jax.numpy import index_exp as joi
from exojax.utils.indexing import getix
from exojax.spec.modit import inc2D_givenx
from exojax.spec.lpf import voigt
from jax import scipy as jsc


@jit
def xsvector(cnu, indexnu, R, nsigmaD, ngammaL, S, nu_grid, ngammaL_grid, qvector):
    """Cross section vector (REDIT version)

    The original code is rundit_fold_logredst in [addit package](https://github.com/HajimeKawahara/addit). DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid
       qvector: kernel length zero array

    Returns:
       Cross section in the linear nu grid
    """
    Ng_nu = len(nu_grid)
    Ng_gammaL = len(ngammaL_grid)

    log_ngammaL = jnp.log(ngammaL)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu, 1)
    lsda = jnp.zeros((len(nu_grid), len(ngammaL_grid)))  # LSD array init
    Slsd = inc2D_givenx(lsda, S, cnu, indexnu, log_ngammaL,
                        log_ngammaL_grid)  # LSD

    al = ngammaL_grid[jnp.newaxis, :]
    Mat = jnp.hstack([al.T, Slsd.T])

    def seqconv(x, arr):
        carry = 0.0
        ngammaL_each = arr[0]
        se = arr[1:]
        kernel = voigt(qvector, nsigmaD, ngammaL_each)
        # arr=jnp.convolve(se,kernel,mode="same")
        arr = jsc.signal.convolve(se, kernel, mode='same')
        return carry, arr

    val, xsmm = scan(seqconv, 0.0, Mat)
    xsm = jnp.sum(xsmm, axis=0)*R/nu_grid

    return xsm


@jit
def xsmatrix(cnu, indexnu, R, nsigmaDl, ngammaLM, SijM, nu_grid, dgm_ngammaL, qvector):
    """Cross section matrix for xsvector (REDIT)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       nu_lines: line center (Nlines)
       nsigmaDl: normalized doppler sigma in layers in R^(Nlayer x 1)
       ngammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_ngammaL: DIT Grid Matrix for normalized gammaL R^(Nlayer, NDITgrid)
       qvector: kernel length zero array

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    NDITgrid = jnp.shape(dgm_ngammaL)[1]
    Nline = len(cnu)
    Mat = jnp.hstack([nsigmaDl, ngammaLM, SijM, dgm_ngammaL])

    def fxs(x, arr):
        carry = 0.0
        nsigmaD = arr[0:1]
        ngammaL = arr[1:Nline+1]
        Sij = arr[Nline+1:2*Nline+1]
        ngammaL_grid = arr[2*Nline+1:2*Nline+NDITgrid+1]
        arr = xsvector(cnu, indexnu, R, nsigmaD, ngammaL,
                       Sij, nu_grid, ngammaL_grid, qvector)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm


if __name__ == '__main__':
    print('test')
