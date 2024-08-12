"""(line) shape filters used in PreSOLAR algorithm
"""

from exojax.spec.lpf import voigt
import jax.numpy as jnp

def generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length):
    """generate a Voigt filter with a tail cut (naturally!)

    Args:
        nsigmaD (float): normalized Dopper width
        ngammaL (float): normalized Lorenz half width
        filter_length (int): filter length

    Returns:
        _type_: _description_
    """
    qogrid = jnp.array(range(-filter_length, filter_length))
    return voigt(qogrid, nsigmaD, ngammaL)


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


def check_filter_condition(shape_filter):
    """_summary_

    Args:
        shape_filter (_type_): _description_
    """
    if np.mod(len(shape_filter), 2) == 0:
        raise ValueError("shape filter length must be odd.")
