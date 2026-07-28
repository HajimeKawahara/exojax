import warnings

import numpy as np
import pytest
import jax.numpy as jnp
from jax import jit
from exojax.utils.grids import wavenumber_grid
from exojax.postproc.response import ipgauss_sampling
from exojax.postproc.response import ipgauss_ola_sampling
from exojax.postproc.response import ipgauss_variable_sampling
from exojax.postproc.response import sampling_band_integral
from exojax.utils.grids import velocity_grid

from exojax.utils.constants import c

def _ipgauss_sampling_naive(nusd, nus, F0, beta, RV):
    """Apply the Gaussian IP response + sampling to a spectrum F.

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)

    Return:
        response-applied spectrum (F)
    """
    #    The following check should be placed as another function.
    #    if(np.min(nusd) < np.min(nus) or np.max(nusd) > np.max(nus)):
    #        print('WARNING: The wavenumber range of the observational grid [', np.min(nusd), '-', np.max(nusd), ' cm^(-1)] is not fully covered by that of the model grid [', np.min(nus), '-', np.max(nus), ' cm^(-1)]. This can result in the incorrect response-applied spectrum. Check the wavenumber grids for the model and observation.', sep='')

    @jit
    def ipgauss_sampling_jax(nusd, nus, F0, beta, RV):
        dvmat = jnp.array(c * jnp.log(nusd[None, :] / nus[:, None]))
        kernel = jnp.exp(-(dvmat + RV)**2 / (2.0 * beta**2))
        kernel = kernel / jnp.sum(kernel, axis=0)  # axis=N
        F = kernel.T @ F0
        return F

    F = ipgauss_sampling_jax(nusd, nus, F0, beta, RV)
    return F

def test_ipgauss_sampling(fig=False):
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")
    F0 = np.ones_like(nus)
    F0[500 - 5:500 + 5] = 0.5
    RV = 10.0
    beta = 20.0
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               250,
                                               xsmode="lpf")
                                               #settings before HMC
    vsini_max = 100.0
    vr_array = velocity_grid(resolution, vsini_max)

    F = ipgauss_sampling(nusd, nus, F0, beta, RV, vr_array)

    F_naive = _ipgauss_sampling_naive(nusd, nus, F0, beta, RV)
    res = np.max(np.abs(1.0 - F_naive/F))
    print(res)
    assert res < 1.e-4 #0.1% allowed
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nusd,F)
        plt.plot(nusd,F_naive,ls="dashed")
        plt.show()

def test_ipgauss_ola_sampling(fig=False):
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               10000,
                                               xsmode="premodit")
    F0 = np.ones_like(nus)
    F0[5000 - 50:5000 + 50] = 0.5
    RV = 10.0
    beta = 2.0
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               2500,
                                               xsmode="lpf")
                                               #settings before HMC
    vsini_max = 10.0
    vr_array = velocity_grid(resolution, vsini_max)

    input_matrix = F0.reshape((5,2000))
    print(jnp.shape(input_matrix),jnp.shape(vr_array))
    F = ipgauss_ola_sampling(nusd, nus, input_matrix, beta, RV, vr_array)

    F_naive = _ipgauss_sampling_naive(nusd, nus, F0, beta, RV)
    res = np.max(np.abs(1.0 - F_naive/F))
    print(res)
    assert res < 3.e-3 
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nusd,F)
        plt.plot(nusd,F_naive,ls="dashed")
        plt.show()

def test_sampling_band_integral(fig=False):
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")
    F0 = 2.0 * wav
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               250,
                                               xsmode="lpf")
                                               #settings before HMC

    logstep = 1. / resolution_inst
    nusd_max = nusd * np.exp(logstep/2.)
    nusd_min = nusd / np.exp(logstep/2.)

    wavd = 1.0e8 / nusd
    wavd_min = 1.0e8 / nusd_max
    wavd_max = 1.0e8 / nusd_min

    F_band_sampling = sampling_band_integral(nus, F0, wavd_min, wavd_max)
    F_ana = (wavd_max**2. - wavd_min**2.) / (wavd_max - wavd_min)

    res = np.max(np.abs(1.0 - F_ana/F_band_sampling))
    print(res)
    assert res < 1.e-4 #0.01% allowed
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nus,F0, '+')
        plt.plot(nusd, F_band_sampling, '+')
        plt.plot(nusd, F_ana, '+')
        plt.show()



def test_SopInstProfile():
    from exojax.postproc.specop import SopInstProfile
    
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")
    F0 = np.ones_like(nus)
    F0[500 - 5:500 + 5] = 0.5
    RV = 10.0
    beta = 20.0
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               250,
                                               xsmode="lpf")
    
    SopInst = SopInstProfile(nus)
    
    F = SopInst.ipgauss(F0, beta)
    F = SopInst.sampling(F, RV, nusd)
    F_naive = _ipgauss_sampling_naive(nusd, nus, F0, beta, RV)
    res = np.max(np.abs(1.0 - F_naive/F))
    print(res)
    assert res < 1.e-4 #0.1% allowed


def test_SopInstProfile_warns_from_vrmax_independent_of_nu_grid():
    from exojax.postproc.specop import SopInstProfile

    nu_grid = np.geomspace(4000.0, 8000.0, 100)
    spectrum = np.ones_like(nu_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sop_inst = SopInstProfile(nu_grid, vrmax=10.0)

    assert np.max(np.abs(np.asarray(sop_inst.vrarray))) > 5.0 * 3.0
    with pytest.warns(UserWarning, match="`vrmax`.*too small"):
        sop_inst.check_vrmax(standard_deviation=3.0)
    with pytest.warns(UserWarning, match="`vrmax`.*too small"):
        sop_inst.ipgauss(spectrum, standard_deviation=3.0)


def test_SopInstProfile_does_not_warn_for_sufficient_vrmax():
    from exojax.postproc.specop import SopInstProfile

    nu_grid = np.geomspace(4000.0, 4000.4, 100)
    spectrum = np.ones_like(nu_grid)
    sop_inst = SopInstProfile(nu_grid, vrmax=10.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sop_inst.ipgauss(spectrum, standard_deviation=2.0)


def test_SopInstProfile_accepts_traced_standard_deviation():
    from exojax.postproc.specop import SopInstProfile

    nu_grid = np.geomspace(4000.0, 4000.4, 100)
    spectrum = np.ones_like(nu_grid)
    sop_inst = SopInstProfile(nu_grid, vrmax=10.0)
    apply_ipgauss = jit(
        lambda standard_deviation: sop_inst.ipgauss(
            spectrum, standard_deviation
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = apply_ipgauss(3.0)

    assert result.shape == spectrum.shape


def test_ipgauss_variable_sampling_using_constant_beta_array(fig=False):
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")
    F0 = np.ones_like(nus)
    F0[500 - 5:500 + 5] = 0.5
    RV = 10.0
    beta = 20.0
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               250,
                                               xsmode="lpf")
                                               #settings before HMC
    beta = 20.0
    beta_variable = np.ones_like(nusd)*beta

    F = ipgauss_variable_sampling(nusd, nus, F0, beta_variable, RV)

    F_naive = _ipgauss_sampling_naive(nusd, nus, F0, beta, RV)
    res = np.max(np.abs(1.0 - F_naive/F))
    print(res)
    assert res < 1.e-4 #0.1% allowed
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nusd,F)
        plt.plot(nusd,F_naive,ls="dashed")
        plt.show()


def test_SopInstProfile_ola(fig=False):
    from exojax.postproc.specop import SopInstProfile
    
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               10000,
                                               xsmode="premodit")
    F0 = np.ones_like(nus)
    F0[5000 - 50:5000 + 50] = 0.5
    RV = 4.0
    beta = 4.0
    nusd, wav, resolution_inst = wavenumber_grid(4003.0,
                                               4007.0,
                                               250,
                                               xsmode="lpf")
    
    SopInst = SopInstProfile(nus, convolution_method="exojax.signal.ola")
    
    F = SopInst.ipgauss(F0, beta)
    F = SopInst.sampling(F, RV, nusd)
    F_naive = _ipgauss_sampling_naive(nusd, nus, F0, beta, RV)
    res = np.max(np.abs(1.0 - F_naive/F))
    print(res)
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nusd,F)
        plt.plot(nusd,F_naive,ls="dashed")
        plt.show()

    assert res < 1.e-3 #0.1% allowed


if __name__ == "__main__":
    #test_ipgauss_sampling(fig=True)
    #test_ipgauss_variable_sampling_using_constant_beta_array(fig=True)
    #test_SopInstProfile()
    #test_ipgauss_ola_sampling(fig=True)
    test_SopInstProfile_ola(fig=True)
    # test_sampling_band_integral(fig=True)
