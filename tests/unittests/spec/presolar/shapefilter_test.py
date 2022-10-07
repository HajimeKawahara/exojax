import numpy as np
from exojax.spec.shapefilter import compute_filter_length
from exojax.spec.shapefilter import generate_voigt_shape_filter
from exojax.spec import normalized_doppler_sigma
from exojax.spec.molinfo import molmass
import pytest

def test_compute_filter_length():
    # example 50cm-1 tail cut at 4000cm-1
    spectral_resolution = 10**6
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    assert filter_length == 25001

from exojax.spec.lpf import voigt
import jax.numpy as jnp
def _generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length):
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

def test_generate_voigt_shape_filter(fig=False):
    spectral_resolution = 10**6
    T = 1500.0
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    nsigmaD = normalized_doppler_sigma(T, molmass("CO"), spectral_resolution)
    ngammaL = nsigmaD*np.array([0.1,0.3,1.0,3.0])
    voigtp = generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length)
    assert np.sum(voigtp[;,0]) == pytest.approx(0.9999432)
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(voigtp)
        plt.show()


"""
def test_generate_voigt_shape_filter(fig=False):
    spectral_resolution = 10**6
    T = 1500.0
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    nsigmaD = normalized_doppler_sigma(T, molmass("CO"), spectral_resolution)
    ngammaL = nsigmaD*np.array([0.1,0.3,1.0,3.0])
    voigtp = generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length)
    #assert np.sum(voigtp[;,0]) == pytest.approx(0.9999432)
    if fig:
        import matplotlib.pyplot as plt
        for i in range(len(ngammaL)):
            plt.plot(voigtp[:,i])
        plt.show()
"""


if __name__ == "__main__":
    #test_compute_filter_length()
    #test_generate_voigt_shape_filter(fig=True)
    test_generate_voigt_shape_filter_multi(fig=True)