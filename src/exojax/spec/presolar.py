"""PreSOLAR Precomputing Shape density and OverLap Add convolution Rxxxx

"""
import jax.numpy as jnp

def calc_xsection_from_lsd(reshaped_Slsd, lineshape_filter, R, nsigmaD, nu_grid,
                           log_ngammaL_grid):
    """Compute cross section from LSD using the OLA algorithm

    Args:
        reshaped_Slsd: reshaped line shape density
        lineshape_filter: 
        R: spectral resolution
        nsigmaD: normaized Gaussian STD
        nu_grid: linear wavenumber grid
        log_gammaL_grid: logarithm of gammaL grid

    Returns:
        Cross section in the log nu grid
    """

    Sbuf = jnp.vstack([reshaped_Slsd, jnp.zeros_like(reshaped_Slsd)])
    fftval = jnp.fft.rfft(Sbuf, axis=0)
    Ng_nu = len(nu_grid)
    fftvalsum = jnp.sum(fftval * vk, axis=(1, ))
    return jnp.fft.irfft(fftvalsum)[:Ng_nu] * R / nu_grid
