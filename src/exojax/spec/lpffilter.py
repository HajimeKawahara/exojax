import jax.numpy as jnp
from exojax.spec.lpf import voigt

def generate_closed_lpffilter(filter_length, nsigmaD, ngammaL):
    """Generates the closed form LPF filter

    Args:
        filter_length (int): length of the wavenumber grid of lpffilter
        nsigmaD (float): normalized gaussian standard deviation, resolution*betaT/nu betaT is the STD of Doppler broadening
        ngammaL (float): normalized Lorentzian half width

    Notes: 
        The filter structure is filter[1:M] = vkfilter[M+1:][::-1]m where M=N/2
        filter[0] is the DC component, Nyquist component.
        filter[M] is the Nyquist component. 

    Returns:
        array: closed lpf filter [2*filter_length]
    """
    # dq is equivalent to resolution*jnp.log(nu_grid) - resolutiona*jnp.log(nu_grid[0]) (+ Nyquist)
    dq = jnp.arange(0, filter_length + 1)
    lpffilter_oneside = voigt(dq, nsigmaD, ngammaL)
    return jnp.concatenate([lpffilter_oneside, lpffilter_oneside[1:-1][::-1]])

def generate_open_lpffilter(nfilter, nsigmaD, ngammaL):
    """Generates the open form LPF filter

    Args:
        nfilter (int): length of the wavenumber grid of lpffilter
        nsigmaD (float): normalized gaussian standard deviation, resolution*betaT/nu betaT is the STD of Doppler broadening
        ngammaL (float): normalized Lorentzian half width

    Returns:
        array: open lpf filter [2*filter_length]
    """
    dq = jnp.arange(-nfilter, nfilter)
    return voigt(dq, nsigmaD, ngammaL)
    
