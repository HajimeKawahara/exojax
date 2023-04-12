import pytest
import numpy as np
from exojax.atm.atmprof import pressure_layer_logspace
from exojax.atm.atmprof import pressure_scale_height


def test_log_pressure_is_constant():
    pressure, dParr, k = pressure_layer_logspace(log_pressure_top=-8.,
                                                 log_pressure_btm=2.,
                                                 nlayer=20,
                                                 mode='ascending',
                                                 numpy=False)

    #check P[n-1] = k P[n]
    assert np.all(np.abs(1.0 - pressure[1:] * k / pressure[:-1]) < 1.e-5)


def test_atmoshperic_height_for_isothermal_with_analytic():
    from exojax.utils.grids import wavenumber_grid
    from exojax.spec.atmrt import ArtTransPure
    from jax.config import config
    config.update("jax_enable_x64", True)
    mu_fid = 28.00863
    T_fid = 500.
    Nx = 100000
    nu_grid, wav, res = wavenumber_grid(22000.0,
                                        26500.0,
                                        Nx,
                                        unit="AA",
                                        xsmode="premodit")
    art = ArtTransPure(nu_grid,
                       pressure_top=1.e-10,
                       pressure_btm=1.e1,
                       nlayer=100)
    Tarr = T_fid * np.ones_like(art.pressure)
    gravity_btm = 2478.57730044555
    radius_btm = 7149200000.0
    mmw = mu_fid * np.ones_like(art.pressure)

    _, normalized_radius_layer, normalized_radius_lower = art.atmosphere_height(
        Tarr, mmw, radius_btm, gravity_btm)

    #theoretical value
    H_btm = pressure_scale_height(gravity_btm, T_fid, mu_fid)
    dq = np.log(art.pressure[-1]) - np.log(art.pressure)
    normalized_radius_theory = (np.exp(H_btm * dq / radius_btm))
    res = 1.0 - (normalized_radius_lower - 1.0) / (normalized_radius_theory - 1.0)
    assert np.all(np.abs(res[:-1]) < 1.e-11)
    
if __name__ == "__main__":
    test_log_pressure_is_constant()
    test_atmoshperic_height_for_isothermal_with_analytic()