import jax.numpy as jnp
from exojax.spec.lpf import voigt

def _close_filter_length(filter_length_oneside):
    return 2*filter_length_oneside

def generate_closed_lpffilter(filter_length_oneside, nsigmaD, ngammaL):
    """Generates the closed form LPF filter

    Args:
        filter_length (int): length of the wavenumber grid of lpffilter
        nsigmaD (float): normalized gaussian standard deviation, resolution*betaT/nu betaT is the STD of Doppler broadening
        ngammaL (float): normalized Lorentzian half width

    Notes: 
        The filter structure is filter[1:M] = vkfilter[M+1:][::-1]m where M=N/2
        filter[0] is the DC component, Nyquist component.
        filter[M] is the Nyquist component. 
        The dimension of the closed lpf filter is even number.
        
    Returns:
        array: closed lpf filter [2*filter_length_oneside]
    """
    # dq is equivalent to resolution*jnp.log(nu_grid) - resolutiona*jnp.log(nu_grid[0]) (+ Nyquist)
    dq = jnp.arange(0, filter_length_oneside + 1)
    lpffilter_oneside = voigt(dq, nsigmaD, ngammaL)
    return jnp.concatenate([lpffilter_oneside, lpffilter_oneside[1:-1][::-1]])

def _open_filter_length(filter_length_oneside):
    return 2*filter_length_oneside+1

def generate_open_lpffilter(filter_length_oneside, nsigmaD, ngammaL):
    """Generates the open form LPF filter

    Notes:
        The dimension of the open lpf filter is odd number to ensure asymmetry. This forces fft_length to be even number when the dimension of signal is even number.

    Args:
        nfilter (int): length of the wavenumber grid of lpffilter
        nsigmaD (float): normalized gaussian standard deviation, resolution*betaT/nu betaT is the STD of Doppler broadening
        ngammaL (float): normalized Lorentzian half width

    Returns:
        array: open lpf filter [2*filter_length_oneside+1]
    """
    dq = jnp.arange(-filter_length_oneside, filter_length_oneside+1)
    return voigt(dq, nsigmaD, ngammaL)
    
