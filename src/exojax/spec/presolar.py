"""PreSOLAR Precomputing Shape density and OverLap Add convolution Rxxxx

"""
import jax.numpy as jnp

def compute_filter_length(wavenumber_halfwidth, representative_wavenumber,
                          spectral_resolution):
    """compute the length of the FIR line shape filter

    Args:
        wavenumber_halfwidth (float): half width at representative wavenumber (cm-1) 
        representative_wavenumber (float): representative wavenumber (cm-1)
        spectral_resolution (float): spectral resolution R0

    Returns:
        int: filter length
        
    Examples:
        from exojax.utils.instfunc import resolution_eslog
        spectral_resolution = resolution_eslog(nu_grid)
        filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    """
    filter_length = 2 * int(spectral_resolution * wavenumber_halfwidth /
                            representative_wavenumber) + 1
    if filter_length < 3:
        raise ValueError("filter_length less than 3")
    return filter_length


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
