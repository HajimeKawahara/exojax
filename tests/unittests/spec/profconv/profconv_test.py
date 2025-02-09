import pytest
from exojax.spec.profconv import calc_xsection_from_lsd_zeroscan
from exojax.spec.profconv import calc_xsection_from_lsd_scanfft
import jax.numpy as jnp
import numpy as np


@pytest.mark.parametrize("i", [0, -1])
def test_basic_convolution_calc_xsection_from_lsd_zeroscan(i):
    Nsignal, Ngamma, Slsd = _sld_sample(i)
    # these are used for log-linear conversuion, so assume identical convolution in this test
    R = 1.0
    nu_grid = jnp.ones(Nsignal)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    nsigmaD = 1.0
    log_ngammaL_grid = jnp.ones(Ngamma)

    xsv = calc_xsection_from_lsd_zeroscan(
        Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
    )
    xsv_ref = _ref_value_xsv(i)
    assert jnp.allclose(xsv, xsv_ref)
    return xsv


@pytest.mark.parametrize("i", [0, -1])
def test_agreement_zeroscan_scanfft(i):
    Nsignal, Ngamma, Slsd = _sld_sample(i)
    # these are used for log-linear conversuion, so assume identical convolution in this test
    R = 1.0
    nu_grid = jnp.ones(Nsignal)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    nsigmaD = 1.0
    log_ngammaL_grid = jnp.ones(Ngamma)

    xsv = calc_xsection_from_lsd_zeroscan(
        Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
    )
    xsv2 = calc_xsection_from_lsd_scanfft(
        Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid
    )
    assert jnp.allclose(xsv, xsv2)


def _sld_sample(i):
    Nsignal = 16
    Ngamma = 2
    Slsd = np.zeros((Nsignal, Ngamma))
    Slsd[i, :] = 1.0
    return Nsignal, Ngamma, Slsd

def _ref_value_xsv(i):
    xsv_ref = np.array(
            [
                0.21061051,
                0.19358274,
                0.15388829,
                0.11205564,
                0.07943292,
                0.05703694,
                0.04214784,
                0.03212026,
                0.02517008,
                0.02020074,
                0.01654267,
                0.01377992,
                0.01164654,
                0.00996912,
                0.00864311,
                0.0076811,
            ]
        )
    if i==0:
        return xsv_ref
    elif i == -1:
        return xsv_ref[::-1]
    else:
        raise ValueError("Invalid i")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xsv = test_basic_convolution_calc_xsection_from_lsd_zeroscan(0)
    xsv = test_basic_convolution_calc_xsection_from_lsd_zeroscan(-1)
    
    # plt.plot(xsv)
    # plt.yscale("log")
    # plt.show()
