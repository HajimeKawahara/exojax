import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from jax.numpy import index_exp
from jax.lax import dynamic_update_slice
from jax import jit

from scipy import fft
from scipy.special import lambertw
import math


def optimal_div_length(filter_length):
    """optimal divided sector length of OLA
    
    Notes:
        This code was taken and modified from scipy.signal._signaltools._oa_calc_oalens
        under BSD 3-Clause "New" or "Revised" License
        
    Args:
        filter_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    overlap = filter_length - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    div_length = fft.next_fast_len(math.ceil(opt_size))

    return div_length


@jit
def olaconv(input_matrix, fir_filter):
    """Overlap and Add convolve (jax.numpy version)

    Args:
        input_matrix (jax.ndarray): reshaped matrix to (ndiv, div_length) of the input 
        fir_filter (jax.ndarray): real FIR filter. The length should be odd.
        
    Note:
        ndiv is the number of the divided input sectors.
        div_length is the length of the divided input sectors. 
        
    Returns:
        convolved vector w/ output length of (len(input vector) + len(fir_filter) - 1)
    """
    ndiv, div_length = jnp.shape(input_matrix)
    filter_length = len(fir_filter)
    fft_length = div_length + filter_length - 1
    input_length = ndiv * div_length
    output_length = input_length + filter_length - 1

    xzeropad = jnp.zeros((ndiv, fft_length))
    xzeropad = xzeropad.at[index_exp[:, 0:div_length]].add(input_matrix)
    fzeropad = jnp.zeros(fft_length)
    fzeropad = fzeropad.at[index_exp[0:filter_length]].add(fir_filter)
    ftilde = jnp.fft.rfft(fzeropad)
    xtilde = jnp.fft.rfft(xzeropad, axis=1)
    ytilde = xtilde * ftilde[jnp.newaxis, :]
    ftarr = jnp.fft.irfft(ytilde, axis=1)
    fftval = overlap_and_add(ftarr, output_length, div_length)
    return fftval


def overlap_and_add(ftarr, output_length, div_length):
    """Compute overlap and add

    Args:
        ftarr (jax.ndarray): filtered input matrix
        output_length (int): length of the output of olaconv
        div_length (int): the length of the divided input sectors, equivalent to block_size in scipy 
        
    Returns:
        overlapped and added vector
    """
    def fir_filter(y_and_idiv, ft):
        y, idiv = y_and_idiv
        yzero = jnp.zeros(output_length)
        y = y + dynamic_update_slice(yzero, ft, (idiv * div_length, ))
        idiv += 1
        return (y, idiv), None

    y = jnp.zeros(output_length)
    fftval_and_nscan, _ = scan(fir_filter, (y, 0), ftarr)
    fftval, nscan = fftval_and_nscan
    return fftval


def np_olaconv(input_matrix, fir_filter):
    """Overlap and Add convolve (numpy version)

    Args:
        input_matrix (jax.ndarray): reshaped matrix to (ndiv, div_length) of the input 
        fir_filter (jax.ndarray): real FIR filter. The length should be odd.
        
    Returns:
        convolved vector w/ length of (len(input) + len(fir_filter) - 1)
    """
    ndiv, div_length = np.shape(input_matrix)
    filter_length = len(fir_filter)
    fft_length = div_length + filter_length - 1
    input_length = ndiv * div_length
    xzeropad = np.zeros((ndiv, fft_length))
    xzeropad[:, 0:div_length] = input_matrix
    fzeropad = np.zeros(fft_length)
    fzeropad[0:filter_length] = fir_filter
    ftilde = np.fft.rfft(fzeropad)
    xtilde = np.fft.rfft(xzeropad, axis=1)
    ytilde = xtilde * ftilde[np.newaxis, :]
    ftarr = np.fft.irfft(ytilde, axis=1)
    y = np.zeros(input_length + filter_length - 1)
    for idiv in range(ndiv):
        y[int(idiv * div_length):int(idiv * div_length +
                                     fft_length)] += ftarr[idiv, :]
    return y
