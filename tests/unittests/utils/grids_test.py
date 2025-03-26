import pytest
import numpy as np
from exojax.utils.grids import extended_wavenumber_grid
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.grids import delta_velocity_from_resolution
from exojax.utils.grids import check_eslog_wavenumber_grid
from exojax.utils.grids import check_grid_mode_in_xsmode
from exojax.utils.checkarray import is_sorted
from exojax.utils.checkarray import is_outside_range



def test_extended_wavenumber_grid():
    Nx = 20
    nus, wav_revert, res = wavenumber_grid(
        1000, 10000, Nx, unit="AA", xsmode="premodit", wavelength_order="ascending"
    )
    nleft = 6
    nright = 7
    nus_ext = extended_wavenumber_grid(nus, nleft, nright)
    lnus_ext = np.log(nus_ext)
    dlognu = lnus_ext[1:]-lnus_ext[:-1]
    assert np.all(dlognu == pytest.approx(1.0/res*np.ones_like(dlognu)))
    

@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_wavenumber_grid_order(order):
    Nx = 4000
    nus, wav_revert, res = wavenumber_grid(
        29200.0, 29300.0, Nx, unit="AA", xsmode="lpf", wavelength_order=order
    )
    assert is_sorted(wav_revert) == order


def test_wavenumber_grid():
    Nx = 4000
    nus, wav_revert, res = wavenumber_grid(
        29200.0, 29300.0, Nx, unit="AA", xsmode="lpf", wavelength_order="ascending"
    )
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
    assert resmax < 3.0 * 1.0e-7


def test_velocity_grid():
    resolution = 10**5
    vmax = 150.0  # km/s
    x = velocity_grid(resolution, vmax)
    x = x / vmax
    assert x[0] <= -1.0 and x[-1] >= 1.0
    assert x[1] >= -1.0 and x[-2] <= 1.0


def test_check_eslog_wavenumber_grid():
    nus, wav, res = wavenumber_grid(
        22999, 23000, 1000, unit="AA", xsmode="modit", wavelength_order="descending"
    )
    assert check_eslog_wavenumber_grid(nus)
    nus, wav, res = wavenumber_grid(
        22999, 23000, 10000, unit="AA", xsmode="modit", wavelength_order="descending"
    )
    assert check_eslog_wavenumber_grid(nus)
    nus, wav, res = wavenumber_grid(
        22999, 23000, 100000, unit="AA", xsmode="modit", wavelength_order="descending"
    )
    assert check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.0e8 / 23000.0, 1.0e8 / 22999.0, 1000)
    assert not check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.0e8 / 23000.0, 1.0e8 / 22999.0, 10000)
    assert not check_eslog_wavenumber_grid(nus)
    nus = np.linspace(1.0e8 / 23000.0, 1.0e8 / 22999.0, 100000)
    assert not check_eslog_wavenumber_grid(nus)


def test_check_scale_xsmode():
    assert check_grid_mode_in_xsmode("lpf") == "ESLOG"
    assert check_grid_mode_in_xsmode("modit") == "ESLOG"
    assert check_grid_mode_in_xsmode("premodit") == "ESLOG"
    assert check_grid_mode_in_xsmode("presolar") == "ESLOG"
    assert check_grid_mode_in_xsmode("dit") == "ESLIN"
    assert check_grid_mode_in_xsmode("LPF") == "ESLOG"
    assert check_grid_mode_in_xsmode("MODIT") == "ESLOG"
    assert check_grid_mode_in_xsmode("PREMODIT") == "ESLOG"
    assert check_grid_mode_in_xsmode("PRESOLAR") == "ESLOG"
    assert check_grid_mode_in_xsmode("DIT") == "ESLIN"


def test_is_sorted():
    import numpy as np

    a = np.array([1, 2, 3])
    b = np.array([1, 3, 2])

    assert is_sorted(a) == "ascending"
    assert is_sorted(a[::-1]) == "descending"
    assert is_sorted(b) == "unordered"
    assert is_sorted(2.0) == "single"


import pytest


@pytest.mark.parametrize(
    "xarr, xs, xe, expected",
    [
        (np.array([1.2, 1.4, 1.7, 1.3, 1.0]), 0.7, 0.8, True),  # No element in range
        (np.array([0.75, 1.0, 1.5]), 0.7, 0.8, False),  # One element in range
        (np.array([0.6, 0.9, 1.2]), 0.7, 0.8, True),  # No element in range
        (np.array([]), 0.5, 1.0, True),  # Empty array
        (np.array([0.5, 1.0]), 0.5, 1.0, True),  # Boundary values
    ],
)
def test_is_outside_range(xarr, xs, xe, expected):
    assert is_outside_range(xarr, xs, xe) == expected




if __name__ == "__main__":
    test_extended_wavenumber_grid()
    #test_optimal_nu_grid_length()