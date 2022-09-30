"""PreSOLAR Precomputing Shape density and OverLap Add convolution Rxxxx

"""
import numpy as np
import jax.numpy as jnp
from exojax.signal.ola import optimal_fft_length


def optimal_mini_batch(input_length, filter_length):
    """compute the optimal number and length of the mini batches array

    Args:
        input_length (int): input length i.e. the length of the wavenumber bin
        filter_length (int): filter length
    
    Returns:
        int, int: the optimal number, length of the mini batches.
    """
    opt_div_length = optimal_fft_length(filter_length) - filter_length + 1
    ndiv = int(input_length / opt_div_length) + 1
    return ndiv, opt_div_length

