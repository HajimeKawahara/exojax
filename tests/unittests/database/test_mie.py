from exojax.database.mie import auto_rgrid
from exojax.database.mie import cubeweighted_integral_checker
from exojax.database.mie import mie_lognormal_pymiescatt
import numpy as np
from scipy import integrate


def test_autogrid():
    """tests robust sigmag range for auto_rgrid. Currently 1.0001,4 is within 1 % for the default setting
    """
    lower_limit_sigmag = 1.0001
    upper_limit_sigmag = 4.0
    rg_um = 0.05  # 0.1um = 100nm
    cm2um = 1.0e4
    cm2nm = 1.0e7
    rg = rg_um / cm2um  # in cgs
    rg_nm = rg * cm2nm
    sigr = np.linspace(lower_limit_sigmag, upper_limit_sigmag, 100)
    for sigmag in sigr:
        rgrid = auto_rgrid(rg_nm, sigmag)
        check = cubeweighted_integral_checker(rgrid, rg_nm, sigmag, accuracy=1.0e-2)
        assert check, f"Grid integration failed for sigmag={sigmag}"


def test_autogrid_narrow_distribution_float32():
    rg_nm = np.float32(50.0)
    sigmag = np.float32(1.0001)

    rgrid = auto_rgrid(rg_nm, sigmag)

    assert cubeweighted_integral_checker(rgrid, rg_nm, sigmag)


def test_cubeweighted_integral_checker():
    rg_um = 0.05  # 0.1um = 100nm
    sigmag = 2.0
    cm2um = 1.0e4
    cm2nm = 1.0e7
    rg = rg_um / cm2um  # in cgs
    rg_nm = rg * cm2nm
    rgrid_lower = 1.0
    rgrid_upper = 10000.0
    nrgrid = 1000
    rgrid = np.linspace(rgrid_lower, rgrid_upper, nrgrid)

    check = cubeweighted_integral_checker(rgrid, rg_nm, sigmag)

    assert check


def test_mie_lognormal_pymiescatt_without_scipy_trapz(monkeypatch):
    monkeypatch.delattr(integrate, "trapz", raising=False)

    coefficients = mie_lognormal_pymiescatt(
        1.5 + 0.01j,
        wavelength=500.0,
        sigmag=2.0,
        rg=100.0,
        N0=1.0,
        rgrid=np.linspace(1.0, 1000.0, 10),
    )

    assert integrate.trapz is integrate.trapezoid
    assert np.shape(coefficients) == (7,)
    assert np.all(np.isfinite(coefficients))
