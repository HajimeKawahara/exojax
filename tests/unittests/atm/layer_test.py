import pytest
import numpy as np
from exojax.atm.atmprof import pressure_layer_logspace
from exojax.atm.atmprof import pressure_upper_logspace
from exojax.atm.atmprof import pressure_lower_logspace
from exojax.atm.atmprof import pressure_scale_height


def test_log_pressure_is_constant():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-8.0,
        log_pressure_btm=2.0,
        nlayer=20,
        mode="ascending",
        numpy=False,
    )

    # check P[n-1] = k P[n]
    assert np.all(np.abs(1.0 - pressure[1:] * k / pressure[:-1]) < 1.0e-5)


def test_pressure_upper_logspace():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-3.0,
        log_pressure_btm=2.0,
        nlayer=6,
        mode="ascending",
        numpy=False,
    )
    p_upper = pressure_upper_logspace(pressure, k)
    ref = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5])
    assert np.all(np.log10(p_upper) - ref < 1.0e-5)


def test_pressure_lower_logspace():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-3.0,
        log_pressure_btm=2.0,
        nlayer=6,
        mode="ascending",
        numpy=False,
    )
    p_lower = pressure_lower_logspace(pressure, k)
    ref = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    assert np.all(np.log10(p_lower) - ref < 1.0e-5)


def test_pressure_scale_height_earth():
    gravity_earth = 980.665  # cm/s2
    T = 288.15  # K
    mu_earth = 28.9644
    H = pressure_scale_height(gravity_earth, T, mu_earth)
    ref = 843465.7516276574  # cm (8.4km)

    assert H == pytest.approx(ref)


def test_atmospheric_scale_height_for_isothermal_with_analytic():
    from exojax.utils.grids import wavenumber_grid
    from exojax.rt import ArtTransPure
    from jax import config

    config.update("jax_enable_x64", True)
    mu_fid = 28.00863
    T_fid = 500.0
    Nx = 100000
    nu_grid, wav, res = wavenumber_grid(
        22000.0, 26500.0, Nx, unit="AA", xsmode="premodit"
    )
    art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0e1, nlayer=100)
    Tarr = T_fid * np.ones_like(art.pressure)
    gravity_btm = 2478.57730044555
    radius_btm = 7149200000.0
    mmw = mu_fid * np.ones_like(art.pressure)

    normalized_height, normalized_radius_lower = art.atmosphere_height(
        Tarr, mmw, radius_btm, gravity_btm
    )

    # theoretical value
    H_btm = pressure_scale_height(gravity_btm, T_fid, mu_fid)
    dq = np.arange(0, len(art.pressure))[::-1] * np.log(
        art.pressure_decrease_rate
    )  # n log(k)
    normalized_radius_theory = 1 / (1 + H_btm / radius_btm * dq)
    res = 1.0 - (normalized_radius_lower - 1.0) / (normalized_radius_theory - 1.0)

    assert np.all(np.abs(res[:-1]) < 1.0e-11)


if __name__ == "__main__":
    test_log_pressure_is_constant()
    test_atmospheric_scale_height_for_isothermal_with_analytic()
    test_pressure_upper_logspace()
    test_pressure_lower_logspace()
    test_pressure_scale_height_earth()
