"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
@jit
def xsvector(cnu, indexnu, R, pmarray, nsigmaD, ngammaL, S, nu_grid, ngammaL_grid):
    """Cross section vector (PreMODIT)

    The original code is rundit_fold_logredst in `addit package <https://github.com/HajimeKawahara/addit>`_ ). MODIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the linear nu grid
    """

    Ng_nu = len(nu_grid)
    Ng_gammaL = len(ngammaL_grid)

    log_nstbeta = jnp.log(nsigmaD)
    log_ngammaL = jnp.log(ngammaL)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu, 1)
    lsd_array = jnp.zeros((len(nu_grid), len(ngammaL_grid)))
    Slsd = inc3D_allgiven(lsd_array, S, cnu, indexnu, log_ngammaL,
                        log_ngammaL_grid)  # Lineshape Density
    Sbuf = jnp.vstack([Slsd, jnp.zeros_like(Slsd)])

    # -----------------------------------------------
    # MODIT w/o new folding
    # til_Voigt=voigt_kernel_logst(k, log_nstbeta,log_ngammaL_grid)
    # til_Slsd = jnp.fft.rfft(Sbuf,axis=0)
    # fftvalsum = jnp.sum(til_Slsd*til_Voigt,axis=(1,))
    # xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nu_grid
    # -----------------------------------------------

    fftval = jnp.fft.rfft(Sbuf, axis=0)
    vmax = Ng_nu
#    vk = fold_voigt_kernel_logst(
#        k, log_nstbeta, log_ngammaL_grid, vmax, pmarray)
    fftvalsum = jnp.sum(fftval*vk, axis=(1,))
    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nu_grid

    return xs

