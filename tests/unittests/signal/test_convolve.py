import numpy as np
import pytest

from exojax.signal.convolve import convolve_same


@pytest.mark.parametrize("input_length", [40, 41])
def test_convolve_same_matches_direct_convolution(input_length):
    input_signal = np.linspace(1.0, 2.0, input_length)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    kernel /= kernel.sum()

    actual = convolve_same(input_signal, kernel)
    expected = np.convolve(input_signal, kernel, mode="same")

    assert actual.shape == input_signal.shape
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
