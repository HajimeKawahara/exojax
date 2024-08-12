import numpy as np
from exojax.experimental.shapefilter import compute_filter_length
from exojax.experimental.shapefilter import generate_voigt_shape_filter
from exojax.spec.hitran import normalized_doppler_sigma
from exojax.spec.molinfo import molmass_isotope
from jax import vmap
import pytest


def test_compute_filter_length():
    # example 50cm-1 tail cut at 4000cm-1
    spectral_resolution = 10**6
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    assert filter_length == 25001


def test_generate_voigt_shape_filter(fig=False):
    spectral_resolution = 10**6
    T = 1500.0
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    nsigmaD = normalized_doppler_sigma(T, molmass_isotope("CO"), spectral_resolution)
    ngammaL = nsigmaD
    voigtp = generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length)
    assert np.sum(voigtp) == pytest.approx(0.9999432)


def test_generate_voigt_shape_filter_vmapped(fig=False):
    """test generate_voigt_shape_filter (vmapped)
    """
    spectral_resolution = 10**6
    T = 1500.0
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    nsigmaD = normalized_doppler_sigma(T, molmass_isotope("CO"), spectral_resolution)
    ngammaL = nsigmaD * np.array([0.1, 0.3, 1.0, 3.0])
    vmap_generate_voigt_shape_filter = vmap(generate_voigt_shape_filter,
                                            (None, 0, None), 0)
    voigtp = vmap_generate_voigt_shape_filter(nsigmaD, ngammaL, filter_length)
    ref = np.array([0.9999944, 0.999983, 0.9999432, 0.9998298])
    assert np.all(np.sum(voigtp, axis=1) == pytest.approx(ref))
    if fig:
        import matplotlib.pyplot as plt
        for i, ngammaL_each in enumerate(ngammaL):
            plt.plot(voigtp[i, :],
                     label="$\\acute{\\gamma}_L=$" + str(ngammaL_each))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #test_compute_filter_length()
    #test_generate_voigt_shape_filter(fig=True)
    test_generate_voigt_shape_filter_vmapped(fig=True)
