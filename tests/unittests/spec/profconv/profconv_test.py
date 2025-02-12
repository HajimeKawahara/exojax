from jax import config
import pytest
from exojax.spec.profconv import calc_xsection_from_lsd_zeroscan
from exojax.spec.profconv import calc_open_xsection_from_lsd_zeroscan
from exojax.spec.profconv import calc_xsection_from_lsd_scanfft
from exojax.utils.grids import extended_wavenumber_grid
from exojax.utils.grids import wavenumber_grid
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



def test_basic_convolution_calc_open_xsection_from_lsd_zeroscan(i):
    config.update("jax_enable_x64", True)
    Nsignal, Ngamma, Slsd = _sld_sample(i)
    # these are used for log-linear conversuion, so assume identical convolution in this test
    R = 1.0
    nu_grid, wav, res = wavenumber_grid(22000, 23000, Nsignal, unit="AA", xsmode="premodit", wavelength_order="descending")
    nsigmaD = 1.0
    log_ngammaL_grid = jnp.ones(Ngamma)
    filter_length_oneside = 15
    nu_grid_extended = extended_wavenumber_grid(nu_grid, filter_length_oneside, filter_length_oneside)

    xsv = calc_open_xsection_from_lsd_zeroscan(
        Slsd, R, nsigmaD, nu_grid_extended, log_ngammaL_grid, filter_length_oneside
    )
    xsv = xsv*nu_grid_extended # avoid the log conversion
    xsv_ref = _ref_value_xsv(-i-1)

    if i==-1:
        #res_alias = np.max(np.abs(xsv[-filter_length_oneside-1:]/xsv_ref[0:filter_length_oneside+1] - 1.0))
        res_alias = np.sum(xsv[-filter_length_oneside-1:]-xsv_ref[0:filter_length_oneside+1])
    
    elif i == 0:
        #res_alias = np.max(np.abs(xsv[1:filter_length_oneside]/xsv_ref[1:filter_length_oneside] - 1.0))
        res_alias = np.sum(xsv[0:filter_length_oneside] - xsv_ref[0:filter_length_oneside])
    else:
        raise ValueError("Invalid i")
    print(res_alias)
        #assert res_alias < 2.5e-4 #0.00023432348053409324
    return xsv, xsv_ref, filter_length_oneside, res_alias


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #xsv = test_basic_convolution_calc_xsection_from_lsd_zeroscan(0)
    #xsv = test_basic_convolution_calc_xsection_from_lsd_zeroscan(-1)
    xsv, xsv_ref,filter_length_oneside, res_alias = test_basic_convolution_calc_open_xsection_from_lsd_zeroscan(0)
    d1=0.00012503326575752718
    d2=0.0001357465803679611
    #because the summation is not preserved....
    filter_sum = _ref_value_xsv(0)
    print(jnp.sum(xsv)+d1+d2)
    print(jnp.sum(filter_sum)+jnp.sum(filter_sum[1:]))
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(xsv[0:filter_length_oneside],"o")
    plt.plot(xsv_ref[0:filter_length_oneside],"+")
    plt.plot(xsv)
    plt.plot(xsv_ref,ls="dashed")
    
    #plt.plot(xsv[-filter_length_oneside-1:],"o")
    #plt.plot(xsv_ref[0:filter_length_oneside+1],"+")
    ax = fig.add_subplot(212)
    plt.plot(xsv[1:filter_length_oneside]/xsv_ref[1:filter_length_oneside] - 1,".")
    #plt.plot(xsv[-filter_length_oneside-1:] - xsv_ref[0:filter_length_oneside+1],".")
    #plt.yscale("log")
    plt.show()
