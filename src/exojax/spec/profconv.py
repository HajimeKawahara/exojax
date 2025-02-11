"""line profile convolution with LSD

    - calc_xsection_from_lsd_zeroscan: compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan
    - calc_xsection_from_lsd_scanfft: compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft (see #277), deprecated

"""

from exojax.spec.ditkernel import fold_voigt_kernel_logst


import jax.numpy as jnp
from jax.lax import scan


def _check_complex(x):
    if x.dtype == jnp.float32:
        return jnp.complex64
    elif x.dtype == jnp.float64:
        return jnp.complex128
    else:
        raise ValueError("Invalid dtype")

def calc_open_xsection_from_lsd_zeroscan(
Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
):
    """Compute open cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan

    Notes:
        The aliasing part is closed and thereby can't be used in OLA.
        #277

    Args:
        Slsd (array): line shape density
        R (float): spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD (float): normaized Gaussian STD
        nu_grid (array): linear wavenumber grid [Nnus]
        log_gammaL_grid (array): logarithm of gammaL grid [Ngamma]

    Returns:
        Open cross section in the log nu grid [Nnus + Nfilter - 1]
    """

def calc_xsection_from_lsd_zeroscan(
    Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
):
    """Compute (closed) cross section array from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan

    Notes:
        The aliasing part is closed and thereby can't be used in OLA.
        #277

    Args:
        Slsd (array): line shape density
        R (float): spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD (float): normaized Gaussian STD
        nu_grid (array): linear wavenumber grid [Nnus]
        log_gammaL_grid (array): logarithm of gammaL grid [Ngamma]

    Returns:
        (Closed) Cross section in the log nu grid [Nnus]
    """

    def f(val, x):
        Slsd_k, vk_k = x
        Slsd_buf_k = jnp.concatenate([Slsd_k, jnp.zeros_like(Slsd_k)])
        ftSlsd_k = jnp.fft.rfft(Slsd_buf_k)
        v = ftSlsd_k * vk_k
        val += v
        return val, None

    Ng_nu = len(nu_grid)
    vk =  
        jnp.log(nsigmaD),
        log_ngammaL_grid,
        

    init = jnp.zeros(vk.shape[0], dtype=_check_complex(vk[0, 0]))
    fftvalvk, _ = scan(f, init, [Slsd.T, vk.T])
    return jnp.fft.irfft(fftvalvk)[:Ng_nu] * R / nu_grid


def calc_xsection_from_lsd_scanfft(
    Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
):
    """Compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft (see #277), deprecated

    Notes:
        This function is deprecated. Use calc_xsection_from_lsd_zeroscan instead.
        The aliasing part is closed and thereby can't be used in OLA.


    Args:
        Slsd: line shape density
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD: normaized Gaussian STD
        nu_grid: linear wavenumber grid
        log_gammaL_grid: logarithm of gammaL grid

    Returns:
        Closed cross section in the log nu grid  [Nnus]
    """

    # add buffer
    Sbuf = jnp.vstack([Slsd, jnp.zeros_like(Slsd)])

    # layer by layer fft
    def f(i, x):
        y = jnp.fft.rfft(x)
        i = i + 1
        return i, y

    nscan, fftval = scan(f, 0, Sbuf.T)
    fftval = fftval.T
    Ng_nu = len(nu_grid)

    # filter kernel
    vk = fold_voigt_kernel_logst(
        jnp.fft.rfftfreq(2 * Ng_nu, 1),
        jnp.log(nsigmaD),
        log_ngammaL_grid,
        Ng_nu,
        pmarray,
    )
    # convolves
    fftvalsum = jnp.sum(fftval * vk, axis=(1,))
    return jnp.fft.irfft(fftvalsum)[:Ng_nu] * R / nu_grid