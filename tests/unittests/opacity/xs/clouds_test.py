from weakref import ref
import numpy as np
import jax.numpy as jnp
import pytest
from exojax.atm import condensate
from exojax.rt.layeropacity import layer_optical_depth_clouds_lognormal
from exojax.rt.layeropacity import single_layer_optical_depth_clouds_lognormal
from exojax.utils.grids import wavenumber_grid


def _setting_test_layer_clouds_dtau():
    N = 1000
    nu_grid, wav, res = wavenumber_grid(
        9000.0, 18000.0, N, xsmode="premodit", unit="AA"
    )
    condensate_substance_density = 7.875  # g/cm^3 Fe
    dParr = np.array([0.01])  # bar
    mmr_condensate = np.array([1.0e-5])  # mmr
    extinction_coefficient = np.ones((1,N)) * 1.0e-11  # beta_0
    rg = 1.0e-5  # cm
    sigmag = 2.0
    gravity = 1.0e5
    ref_value = 0.034889877
    return (
        nu_grid,
        dParr,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
        ref_value,
    )


def test_layer_optical_depth_clouds_lognormal():

    (
        _,
        dParr,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
        ref_value,
    ) = _setting_test_layer_clouds_dtau()

    dtau = layer_optical_depth_clouds_lognormal(
        dParr,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
        N0=1.0,
    )

    assert dtau.shape == (1, 1000)
    assert np.sum(dtau) == pytest.approx(ref_value)

def test_single_layer_optical_depth_clouds_lognormal():

    (
        _,
        dParr,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
        ref_value,
    ) = _setting_test_layer_clouds_dtau()

    dtau = single_layer_optical_depth_clouds_lognormal(
        dParr[0],
        extinction_coefficient[0,:],
        condensate_substance_density,
        mmr_condensate[0],
        rg,
        sigmag,
        gravity,
        N0=1.0,
    )

    assert dtau.shape == (1000,)
    assert np.sum(dtau) == pytest.approx(ref_value)


if __name__ == "__main__":
    test_layer_optical_depth_clouds_lognormal()
    test_single_layer_optical_depth_clouds_lognormal()
