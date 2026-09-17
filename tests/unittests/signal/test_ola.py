"""Overlap-add convolution and padding tests."""

from scipy.signal import oaconvolve
import numpy as np
import pytest
import jax.numpy as jnp

from exojax.signal.ola import (
    generate_padding_matrix,
    generate_zeropad,
    np_olaconv,
    ola_lengths,
    olaconv,
    optimal_fft_length,
    overlap_and_add,
    overlap_and_add_matrix,
)


@pytest.mark.parametrize("filter_length, expected", [(10, 54), (1000000, 18432000)])
def test_optimal_fft_length(filter_length, expected):
    assert optimal_fft_length(filter_length) == expected


@pytest.mark.parametrize("shape", [(2, 4), (2, 4, 2, 3)])
def test_generate_padding_matrix(shape):
    inputs = np.arange(np.prod(shape), dtype=float).reshape(shape)
    padded = generate_padding_matrix(-np.inf, inputs, filter_length=3)

    assert padded.shape == (shape[0], shape[1] + 2, *shape[2:])
    np.testing.assert_array_equal(padded[:, :shape[1]], inputs)
    assert np.all(np.isneginf(padded[:, shape[1]:]))


def test_generate_zeropad():
    inputs = jnp.arange(8.0).reshape(2, 4)
    fir_filter = jnp.array([0.25, 0.5, 0.25])
    padded, filter_padded = generate_zeropad(inputs, fir_filter)

    assert padded.shape == (2, 6)
    assert filter_padded.shape == (6,)
    np.testing.assert_array_equal(padded[:, :4], inputs)
    np.testing.assert_array_equal(padded[:, 4:], 0.0)
    np.testing.assert_array_equal(filter_padded[:3], fir_filter)
    np.testing.assert_array_equal(filter_padded[3:], 0.0)


def _gendata():
    rng = np.random.RandomState(1)
    Nx = 100000
    x = np.zeros(Nx)
    Npulse = 50
    x[rng.choice(list(range(Nx)), Npulse)] = 1.0
    Nf = 301  # odd
    g = np.linspace(-3, 3, Nf)
    f = jnp.array(np.exp(-g * g / 2.0) / np.sqrt(2 * np.pi))  # FIR filter

    ndiv = 100
    xarr = jnp.array(x.reshape(ndiv, int(Nx / ndiv)))
    return x, f, xarr


def test_olaconv():
    x, f, xarr = _gendata()
    oac = oaconvolve(x, f)  # length = Nx + M -1
    ndiv, div_length, filter_length = ola_lengths(xarr, f)
    xarr_hat, f_hat = generate_zeropad(xarr, f)
    ola = olaconv(xarr_hat, f_hat, ndiv, div_length, filter_length)
    maxresidual = np.max(np.sqrt((oac - ola) ** 2))
    assert maxresidual < 1.0e-9  # fp64
    # assert maxresidual < 1.e-6 #fp32


def test_np_olaconv():
    rng = np.random.RandomState(1)
    Nx = 100000
    x = np.zeros(Nx)

    Npulse = 50
    x[rng.choice(list(range(Nx)), Npulse)] = 1.0
    Nf = 301  # odd
    g = np.linspace(-3, 3, Nf)
    f = np.exp(-g * g / 2.0) / np.sqrt(2 * np.pi)  # FIR filter

    oac = oaconvolve(x, f)  # length = Nx + M -1
    ndiv = 100
    xarr = x.reshape(ndiv, int(Nx / ndiv))
    ola = np_olaconv(xarr, f)
    maxresidual = np.max(np.sqrt((oac - ola) ** 2))
    assert maxresidual < 1.0e-15


def test_overlap_and_add():
    ndiv = 2
    div_length = 6  # L
    filter_length = 3  # M
    output_length = ndiv * div_length + filter_length - 1
    ftarr = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
    expected_result = jnp.array(
        [1, 2, 3, 4, 5, 6, 8, 10, 3, 4, 5, 6, 7, 8], dtype=float
    )

    result = overlap_and_add(ftarr, output_length, div_length)

    assert jnp.allclose(result, expected_result)


def test_overlap_and_add_matrix():
    """
    Test the overlap_and_add_matrix function to ensure it correctly combines
    overlapping segments of a matrix into a single output array.
    """
    ndiv = 2
    div_length = 6  # L
    filter_length = 3  # M
    output_length = ndiv * div_length + filter_length - 1
    element = [1, 2, 3, 4, 5, 6, 7, 8] #nlayer = 3
    tri_element = [element, element, element]
    ftarr = jnp.array([tri_element, tri_element], dtype=float)
    # print(ftarr.shape) # (2,3,8) = (ndiv, nlayer, div_length)
    arr = [1, 2, 3, 4, 5, 6, 8, 10, 3, 4, 5, 6, 7, 8]
    expected_result = jnp.array([arr, arr, arr], dtype=float)

    result = overlap_and_add_matrix(ftarr, output_length, div_length)

    assert jnp.allclose(result, expected_result)
