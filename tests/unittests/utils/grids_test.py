import pytest
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.grids import delta_velocity_from_resolution
from exojax.utils.grids import check_eslog_wavenumber_grid
from exojax.utils.grids import check_scale_xsmode


def test_wavenumber_grid():
    Nx = 4000
    nus, wav, res = wavenumber_grid(29200.0,
                                    29300.,
                                    Nx,
                                    unit='AA',
                                    xsmode="lpf")
    dif = np.log(nus[1:]) - np.log(nus[:-1])
    refval = 8.54915417e-07
    assert np.all(dif == pytest.approx(refval * np.ones_like(dif)))


def test_delta_velocity_from_resolution():
    from exojax.utils.constants import c
    N = 60
    Rarray = np.logspace(2, 7, N)
    dv_np = c * np.log1p(1.0 / Rarray)
    dv = delta_velocity_from_resolution(Rarray)
    resmax = np.max(np.abs(dv / dv_np) - 1)
    assert resmax < 3. * 1.e-7


def test_velocity_grid():
    resolution = 10**5
    vmax = 150.0  #km/s
    x = velocity_grid(resolution, vmax)
    x = x / vmax
    assert x[0] <= -1.0 and x[-1] >= 1.0
    assert x[1] >= -1.0 and x[-2] <= 1.0


def test_check_eslog_wavenumber_grid():
    nus, wav, res = wavenumber_grid(22999, 23000, 1000, unit='AA', xsmode="modit")
    assert check_eslog_wavenumber_grid(nus)
    nus, wav, res = wavenumber_grid(22999, 23000, 10000, unit='AA', xsmode="modit")
    assert check_eslog_wavenumber_grid(nus)
    nus, wav, res = wavenumber_grid(22999, 23000, 100000, unit='AA', xsmode="modit")
    assert check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.e8 / 23000., 1.e8 / 22999., 1000)
    assert not check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.e8 / 23000., 1.e8 / 22999., 10000)
    assert not check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.e8 / 23000., 1.e8 / 22999., 100000)
    assert not check_eslog_wavenumber_grid(nus)


def test_check_scale_xsmode():
    assert check_scale_xsmode("lpf") == "ESLOG"
    assert check_scale_xsmode("modit") == "ESLOG"
    assert check_scale_xsmode("premodit") == "ESLOG"
    assert check_scale_xsmode("presolar") == "ESLOG"
    assert check_scale_xsmode("dit") == "ESLIN"
    assert check_scale_xsmode("LPF") == "ESLOG"
    assert check_scale_xsmode("MODIT") == "ESLOG"
    assert check_scale_xsmode("PREMODIT") == "ESLOG"
    assert check_scale_xsmode("PRESOLAR") == "ESLOG"
    assert check_scale_xsmode("DIT") == "ESLIN"


if __name__ == "__main__":
    test_wavenumber_grid()
    test_delta_velocity_from_resolution()
    test_velocity_grid()
