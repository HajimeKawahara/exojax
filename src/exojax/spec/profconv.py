"""line profile convolution with LSD

    - calc_open_xsection_from_lsd_zeroscan: compute open cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan
    - calc_xsection_from_lsd_zeroscan: compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan
    - calc_xsection_from_lsd_scanfft: compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft (see #277), deprecated

"""

from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.lpffilter import generate_open_lpffilter
from exojax.signal.ola import _fft_length
from exojax.spec.lpffilter import _open_filter_length
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap


def _check_complex(x):
    if x.dtype == jnp.float32:
        return jnp.complex64
    elif x.dtype == jnp.float64:
        return jnp.complex128
    else:
        raise ValueError("Invalid dtype")


def calc_open_nu_xsection_from_lsd_zeroscan(
    Slsd, R, nsigmaD, log_ngammaL_grid, filter_length_oneside
):
    """Compute (wavenumber x open cross section9 from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan

    Notes:
        The aliasing part is closed and thereby can't be used in OLA.
        #277
        Why the output is not cross section, but (nu_grid_extended x Cross section)?
        This is related to the scan in OpaPremoditStitch.   
        I (@HajimeKawahara) did not want to use wavenumber_grid_extended in the scan in OpaPremoditStitch.xsvector/xsmatrix 2/28 2025
    


    Args:
        Slsd (array): line shape density [Nnus, Ngamma]
        R (float): spectral resolution
        nsigmaD (float): normaized Gaussian STD
        log_gammaL_grid (array): logarithm of gammaL grid [Ngamma]
        filter_length_oneside (int): one side length of the wavenumber grid of lpffilter, Nfilter = 2*filter_length_oneside + 1

    Returns:
        Open (nu_grid_extended x Cross section) in the log nu grid [Nnus] 
        
    """

    div_length = Slsd.shape[0]
    filter_length = _open_filter_length(filter_length_oneside)
    fft_length = _fft_length(div_length, filter_length)

    def f(val, x):
        Slsd_k, lpffilter_k = x
        Slsd_buf_k = jnp.concatenate([Slsd_k, jnp.zeros(filter_length - 1)])
        ftSlsd_k = jnp.fft.rfft(Slsd_buf_k)
        lpffilter_buf_k = jnp.concatenate([lpffilter_k, jnp.zeros(div_length - 1)])
        vk_k = jnp.fft.rfft(lpffilter_buf_k)
        v = ftSlsd_k * vk_k
        val + v
        val += v
        return val, None

    ngammaL_grid = jnp.exp(log_ngammaL_grid)
    vmap_generate_lpffilter = vmap(generate_open_lpffilter, (None, None, 0), 0)
    lpffilter = vmap_generate_lpffilter(filter_length_oneside, nsigmaD, ngammaL_grid)
    init = jnp.zeros(int(fft_length/2)+1, dtype=_check_complex(lpffilter[0, 0]))
    fftvalvk, _ = scan(f, init, [Slsd.T, lpffilter])
    return  jnp.fft.irfft(fftvalvk) * R 
    
def calc_xsection_from_lsd_zeroscan(
    Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
):
    """Compute cross section from LSD in MODIT algorithm using scan+fft to avoid 4GB memory limit in fft and zero padding in scan

    Notes:
        The aliasing part is closed and thereby can't be used in OLA.
        #277

    Args:
        Slsd (array): line shape density [Nnus, Ngamma]
        R (float): spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD (float): normaized Gaussian STD
        nu_grid (array): wavenumber grid [Nnus]
        log_gammaL_grid (array): logarithm of gammaL grid [Ngamma]

    Returns:
        Closed Cross section in the log nu grid [Nnus]
    """

    def f(val, x):
        Slsd_k, vk_k = x
        Slsd_buf_k = jnp.concatenate([Slsd_k, jnp.zeros_like(Slsd_k)])
        ftSlsd_k = jnp.fft.rfft(Slsd_buf_k)
        v = ftSlsd_k * vk_k
        val += v
        return val, None

    Ng_nu = len(nu_grid)
    vk = fold_voigt_kernel_logst(
        jnp.fft.rfftfreq(2 * Ng_nu, 1),
        nsigmaD,
        log_ngammaL_grid,
        Ng_nu,
        pmarray,
    )

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
        Slsd (array): line shape density [Nnus, Ngamma]
        R (float): spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD (float): normaized Gaussian STD
        nu_grid (array): wavenumber grid [Nnus]
        log_gammaL_grid (array): logarithm of gammaL grid [Ngamma]

    Returns:
        Closed Cross section in the log nu grid [Mnus]
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
        nsigmaD,
        log_ngammaL_grid,
        Ng_nu,
        pmarray,
    )
    # convolves
    fftvalsum = jnp.sum(fftval * vk, axis=(1,))
    return jnp.fft.irfft(fftvalsum)[:Ng_nu] * R / nu_grid
