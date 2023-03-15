import jax.numpy as jnp

def convolve_same(input_signal, kernel):
    """convolve same

    Note:
        this function is substitute of jnp.convolve, which requires cuDNN

    Args:
        input_signal (1D array): original input signal
        kernel (1D array): convolution kernel

    Returns:
        1D array: convolved signal
    """
    input_length = len(input_signal)
    filter_length = len(kernel)
    fft_length = input_length + filter_length - 1
    convolved_signal = jnp.fft.irfft(
        jnp.fft.rfft(input_signal, n=fft_length) * jnp.fft.rfft(kernel, n=fft_length))
    n = int((filter_length - 1) / 2) 
    convolved_signal = convolved_signal[n:-n]
    return convolved_signal

