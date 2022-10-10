"""Line profile computation using Discrete Integral Transform using scan+fft (#277).

* MODIT using scan+fft allows > 4GB fft memory. but approximately 2 times slower than modit.
* When you use modit and get the error such as "failed to initialize batched cufft plan with customized allocator: Allocating 8000000160 bytes exceeds the memory limit of 4294967296 bytes.", 
you should consider to modit_scanfft

"""
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.lsd import inc2D_givenx

def calc_xsection_from_lsd_scanfft(Slsd, R, pmarray, nsigmaD, nu_grid,
                                   log_ngammaL_grid):
    """Compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft (see #277)

    Args:
        Slsd: line shape density
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD: normaized Gaussian STD
        nu_grid: linear wavenumber grid
        log_gammaL_grid: logarithm of gammaL grid

    Returns:
        Cross section in the log nu grid
    """

    Sbuf = jnp.vstack([Slsd, jnp.zeros_like(Slsd)])

    def f(i, x):
        y = jnp.fft.rfft(x)
        i = i + 1
        return i, y

    nscan, fftval = scan(f, 0, Sbuf.T)
    fftval = fftval.T
    Ng_nu = len(nu_grid)
    vk = fold_voigt_kernel_logst(jnp.fft.rfftfreq(2 * Ng_nu, 1),
                                 jnp.log(nsigmaD), log_ngammaL_grid, Ng_nu,
                                 pmarray)
    fftvalsum = jnp.sum(fftval * vk, axis=(1, ))
    return jnp.fft.irfft(fftvalsum)[:Ng_nu] * R / nu_grid



@jit
def xsvector_scanfft(cnu, indexnu, R, pmarray, nsigmaD, ngammaL, S, nu_grid,
             ngammaL_grid):
    """Cross section vector (MODIT scanfft)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nsigmaD: normaized Gaussian STD 
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the log nu grid
    """

    log_ngammaL_grid = jnp.log(ngammaL_grid)
    lsd_array = jnp.zeros((len(nu_grid), len(ngammaL_grid)))
    Slsd = inc2D_givenx(lsd_array, S, cnu, indexnu, jnp.log(ngammaL),
                        log_ngammaL_grid)
    xs = calc_xsection_from_lsd_scanfft(Slsd, R, pmarray, nsigmaD, nu_grid,
                                log_ngammaL_grid)
    return xs


@jit
def xsmatrix_scanfft(cnu, indexnu, R, pmarray, nsigmaDl, ngammaLM, SijM, nu_grid,
             dgm_ngammaL):
    """Cross section matrix for xsvector (MODIT), scan+fft

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nu_lines: line center (Nlines)
       nsigmaDl: normalized doppler sigma in layers in R^(Nlayer x 1)
       ngammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_ngammaL: DIT Grid Matrix for normalized gammaL R^(Nlayer, NDITgrid)

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    NDITgrid = jnp.shape(dgm_ngammaL)[1]
    Nline = len(cnu)
    Mat = jnp.hstack([nsigmaDl, ngammaLM, SijM, dgm_ngammaL])

    def fxs(x, arr):
        carry = 0.0
        nsigmaD = arr[0:1]
        ngammaL = arr[1:Nline + 1]
        Sij = arr[Nline + 1:2 * Nline + 1]
        ngammaL_grid = arr[2 * Nline + 1:2 * Nline + NDITgrid + 1]
        arr = xsvector_scanfft(cnu, indexnu, R, pmarray, nsigmaD, ngammaL, Sij,
                       nu_grid, ngammaL_grid)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm


@jit
def xsmatrix_vald_scanfft(cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS,
                  nu_grid, dgm_ngammaLS):
    """Cross section matrix for xsvector (MODIT) with scan+fft, for VALD lines (asdb)

    Args:
        cnuS: contribution by npgetix for wavenumber [N_species x N_line]
        indexnuS: index by npgetix for wavenumber [N_species x N_line]
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaDlS: normalized sigmaD matrix [N_species x N_layer x 1]
        ngammaLMS: normalized gammaL matrix [N_species x N_layer x N_line]
        SijMS: line intensity matrix [N_species x N_layer x N_line]
        nu_grid: linear wavenumber grid
        dgm_ngammaLS: DIT Grid Matrix (dgm) of normalized gammaL [N_species x N_layer x N_DITgrid]

    Return:
        xsmS: cross section matrix [N_species x N_layer x N_wav]
    """
    xsmS = jit(vmap(xsmatrix_scanfft, (0, 0, None, None, 0, 0, 0, None, 0)))(\
                    cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS, nu_grid, dgm_ngammaLS)
    xsmS = jnp.abs(xsmS)
    return xsmS
