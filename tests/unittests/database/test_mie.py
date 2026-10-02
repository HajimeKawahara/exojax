from exojax.database.mie import auto_rgrid
from exojax.database.mie import compute_mie_coeff_lognormal_grid
from exojax.database.mie import compute_mieparams_cgs_from_miegrid
from exojax.database.mie import cubeweighted_integral_checker
from exojax.database.mie import mie_lognormal
from exojax.database.mie import mie_lognormal_pymiescatt
from exojax.database.mie import read_miegrid
from exojax.database.mie import save_miegrid
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "m,nmedium,expected",
    [
        (
            1.5 + 0.0j,
            1.0,
            [
                0.10433721487271888, 0.10433721487271888, 0.0,
                0.6746057174289772, 0.033950733178967,
                0.02629433936902431, 0.01970075797334274,
            ],
        ),
        (
            1.5 + 0.01j,
            1.0,
            [
                0.10426828014059518, 0.09869338749723963,
                0.00557489264335555, 0.6824829448345409,
                0.03691172640578262, 0.02073235340557222,
                0.01824335214960235,
            ],
        ),
        (
            1.8 + 0.6j,
            1.0,
            [
                0.14204851631761134, 0.06207419535767481,
                0.07997432095993653, 0.646367629054785,
                0.10192576583878754, 0.01012142682617339,
                0.01329549427767259,
            ],
        ),
        (
            1.5 + 0.01j,
            1.33,
            [
                0.022144223687586673, 0.01831618677312923,
                0.0038280369144574437, 0.8607551213528645,
                0.0063784721189600914, 0.0006582519120586556,
                0.008359260794252888,
            ],
        ),
    ],
    ids=["nonabsorbing", "weakly_absorbing", "strongly_absorbing", "medium"],
)
def test_mie_lognormal_reference(m, nmedium, expected):
    """Preserve PyMieScatt 1.8.1.1 Mie_SD reference values without importing it.

    References use SMPS=False, a NumPy lognormal PDF on the diameter grid,
    and m/nmedium and wavelength/nmedium. All radii avoid Rayleigh switching.
    The seven columns include scattering-weighted G and the legacy Bratio
    integral, which is not Bback/Bsca.
    """
    actual = mie_lognormal(
        m, 550.0, 1.7, 100.0, 1.0, np.geomspace(10.0, 1500.0, 128),
        nMedium=nmedium,
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=1.0e-12)


def test_mie_lognormal_density_scaling():
    rgrid = np.geomspace(10.0, 1500.0, 128)
    reference = np.asarray(mie_lognormal(1.5 + 0.01j, 550.0, 1.7, 100.0, 1.0, rgrid))
    actual = mie_lognormal(1.5 + 0.01j, 550.0, 1.7, 100.0, 3.0, rgrid)
    expected = reference * 3.0
    expected[3] = reference[3]

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6)


def test_mie_lognormal_matching_medium():
    actual = mie_lognormal(
        1.33 + 0.0j, 550.0, 1.7, 100.0, 1.0,
        np.geomspace(10.0, 1500.0, 128), nMedium=1.33,
    )

    np.testing.assert_array_equal(actual, np.zeros(7))


def test_mie_lognormal_zero_density():
    actual = mie_lognormal(
        1.5 + 0.01j, 550.0, 1.7, 100.0, 0.0,
        np.geomspace(10.0, 1500.0, 128),
    )

    np.testing.assert_array_equal(actual, np.zeros(7))


def test_generated_miegrid_roundtrip_at_reference_node(monkeypatch, tmp_path):
    rg_arr = np.array([1.0e-5, 2.0e-5])
    sigmag_arr = np.array([1.7, 2.0])
    n0 = 3.0
    monkeypatch.setattr(
        "exojax.database.mie.auto_rgrid",
        lambda rg, sigmag: np.geomspace(10.0, 1500.0, 128),
    )
    miegrid = compute_mie_coeff_lognormal_grid(
        np.array([1.5 + 0.01j]), np.array([550.0]), sigmag_arr, rg_arr, N0=n0,
    )
    filename = tmp_path / "miegrid.npz"
    save_miegrid(filename, miegrid, rg_arr, sigmag_arr)
    stored_grid, stored_rg, stored_sigmag = read_miegrid(filename)

    assert stored_grid.shape == (2, 2, 1, 7)
    np.testing.assert_allclose(stored_grid, miegrid)
    np.testing.assert_allclose(stored_rg, rg_arr)
    np.testing.assert_allclose(stored_sigmag, sigmag_arr)
    actual = compute_mieparams_cgs_from_miegrid(
        rg_arr[0], sigmag_arr[0], stored_grid, stored_rg, stored_sigmag, n0,
    )

    # Independent PyMieScatt reference: cm^2 cross sections and dimensionless g.
    expected = [[1.0426828014059518e-9], [9.869338749723963e-10], [0.6824829448345409]]
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=0.0)


def test_mie_lognormal_legacy_alias_without_pymiescatt(monkeypatch):
    import sys

    monkeypatch.delattr(integrate, "trapz", raising=False)
    monkeypatch.setitem(sys.modules, "PyMieScatt", None)
    monkeypatch.setitem(sys.modules, "PyMieScatt.Mie", None)
    args = (1.5 + 0.01j, 550.0, 1.7, 100.0, 1.0, np.geomspace(10.0, 1500.0, 128))
    expected = mie_lognormal(*args, nMedium=1.33)

    with pytest.warns(DeprecationWarning, match="mie_lognormal"):
        actual = mie_lognormal_pymiescatt(*args, nMedium=1.33)

    np.testing.assert_array_equal(actual, expected)
    assert not hasattr(integrate, "trapz")
