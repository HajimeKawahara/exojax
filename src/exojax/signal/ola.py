import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from jax.numpy import index_exp
from jax.lax import dynamic_update_slice
from jax import jit


@jit
def olaconv(reshaped_input_matrix, fir_filter):
    """Overlap and Add convolve (jax.numpy version)

    Args:
        reshaped_input_matrix (jax.ndarray): reshaped matrix of a long real input vector (ndiv, div_length)
        fir_filter (jax.ndarray): real FIR filter. The length should be odd.
        
    Note:
        ndiv is the number of the divided input vectors.
        div_length is the length of the divided input vectors. 
        
    Returns:
        convolved vector w/ the length of (len(input) + len(f) - 1)
    """
    ndiv, div_length = jnp.shape(reshaped_input_matrix)
    filter_length = len(fir_filter)
    fft_length = div_length + filter_length - 1
    input_length = ndiv * div_length
    xzeropad = jnp.zeros((ndiv, fft_length))
    xzeropad = xzeropad.at[index_exp[:,
                                     0:div_length]].add(reshaped_input_matrix)
    fzeropad = jnp.zeros(fft_length)
    fzeropad = fzeropad.at[index_exp[0:filter_length]].add(fir_filter)
    ftilde = jnp.fft.rfft(fzeropad)
    xtilde = jnp.fft.rfft(xzeropad, axis=1)
    ytilde = xtilde * ftilde[jnp.newaxis, :]
    ftarr = jnp.fft.irfft(ytilde, axis=1)
    fftval = overlap_and_add(ftarr, input_length, filter_length, div_length)
    return fftval


def overlap_and_add(ftarr, input_length, filter_length, div_length):
    def fir_filter(y_and_idiv, ft):
        y, idiv = y_and_idiv
        idiv = idiv + 1
        yzero = jnp.zeros(input_length + filter_length - 1)
        y = y + dynamic_update_slice(yzero, ft, ((idiv - 1) * div_length, ))
        return (y, idiv), None

    y = jnp.zeros(input_length + filter_length - 1)
    fftval_and_nscan, _ = scan(fir_filter, (y, 0), ftarr)
    fftval, nscan = fftval_and_nscan
    return fftval


def np_olaconv(reshaped_input_matrix, fir_filter):
    """Overlap and Add convolve (numpy version)

    Args:
        reshaped_input_matrix (ndarray): reshaped matrix of a long real vector (ndiv, L)
        fir_filter (ndarray): real FIR filter, length should be odd
        
    Returns:
        convolved vector w/ length of (len(input) + len(fir_filter) - 1)
    """
    ndiv, div_length = np.shape(reshaped_input_matrix)
    filter_length = len(fir_filter)
    fft_length = div_length + filter_length - 1
    input_length = ndiv * div_length
    xzeropad = np.zeros((ndiv, fft_length))
    xzeropad[:, 0:div_length] = reshaped_input_matrix
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
