"""PreSOLAR Precomputing Shape density and OverLap Add convolution Rxxxx

    * LBD -> hat(LBD)
    * shapefilter -> hat(shapefilter)

"""
import numpy as np
import jax.numpy as jnp
from exojax.signal.ola import optimal_fft_length
from exojax.signal.ola import generate_padding_matrix


def lbd_olaform(lbd, ndiv, div_length, filter_length):
    """convert LBD to match the form of OLA, i.e. generate hat LBD 

    Args:
        lbd (3D array): line basis density (input length,:,:)
        ndiv (int): number of mini batches
        div_length (int): mini batch length 
        filter_length (int): filter length

    Returns:
        4D array: hat(LBD) (ndiv, fft_length, :, :)
    """
    rlbd = reshape_lbd(lbd, ndiv, div_length)
    hat_lbd = generate_padding_matrix(-np.inf, rlbd, filter_length)
    return hat_lbd


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


def reshape_lbd(lbd, ndiv, div_length, padding_value=-np.inf):
    """reshaping LBD (input length,:,:) to (ndiv, div_length, :, :) w/ padding_value
    Args:
        lbd (3D array): line basis density (input length,:,:)
        ndiv (int): number of mini batches
        div_length (int): mini batch length 
        padding_value (optional): padding value. Defaults to -np.inf.

    Raises:
        ValueError: ndiv*div_length should be larger than input length = shape(lbd)[0]

    Returns:
        4D array: reshaped LBD (ndiv, div_length, :, :) w/ padding_value
    """

    input_length, n_broadening_k, n_E_h = np.shape(lbd)
    if ndiv * div_length < input_length:
        raise ValueError(
            "ndiv*div_length should be larger than input length = shape(lbd)[0]"
        )
    residual = ndiv * div_length - input_length
    padding_shape = (residual, n_broadening_k, n_E_h)
    padding_matrix = np.full(padding_shape, padding_value)
    rlbd = np.vstack((lbd, padding_matrix))
    return rlbd.reshape((ndiv, div_length, n_broadening_k, n_E_h))
