from exojax.database.core.abscoeff import interp_logacia_matrix
from exojax.database.core.abscoeff import interp_logacia_vector
from exojax.database.cia.io import read_cia
from exojax.opacity import OpaCIA
from exojax.test.data import TESTDATA_H2_H2_CIA
from exojax.utils.grids import wavenumber_grid
from importlib.resources import files
import jax
import numpy as np
import pytest


def test_interp_logacia_matrix():
    nus = 4310.0
    nue = 4390.0
    filename = files("exojax").joinpath("data/testdata/" + TESTDATA_H2_H2_CIA)
    nucia, tcia, ac = read_cia(str(filename), nus, nue)
    Tarr = np.array([1000.0, 2000.0])
    logac = np.log10(ac)
    nu_grid, wav, r = wavenumber_grid(nus, nue, 10000, xsmode="premodit")
    logac_cia = interp_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac)
    assert np.all(np.shape(logac_cia) == (2, 10000))
    expected = np.log10(np.interp(nu_grid, nucia, ac[0]))
    np.testing.assert_allclose(logac_cia, np.broadcast_to(expected, (2, 10000)))
    legacy = interp_logacia_matrix(
        Tarr,
        nu_grid,
        nucia,
        tcia,
        logac,
        wavenumber_interpolation="digitize",
    )
    assert np.sum(legacy) == pytest.approx(-891133.44)


def test_interp_logacia_vector():
    nus = 4310.0
    nue = 4390.0
    filename = files("exojax").joinpath("data/testdata/" + TESTDATA_H2_H2_CIA)
    nucia, tcia, ac = read_cia(str(filename), nus, nue)
    T = 2000.0
    logac = np.log10(ac)
    nu_grid, wav, r = wavenumber_grid(nus, nue, 10000, xsmode="premodit")
    logac_cia = interp_logacia_vector(T, nu_grid, nucia, tcia, logac)
    assert np.all(np.shape(logac_cia) == (10000,))
    expected = np.log10(np.interp(nu_grid, nucia, ac[0]))
    np.testing.assert_allclose(logac_cia, expected)


def test_interp_logacia_uses_linear_coefficients_on_both_axes():
    tcia = np.array([200.0, 400.0], dtype=np.float32)
    nucia = np.array([1000.0, 2000.0], dtype=np.float32)
    coefficient = np.array(
        [
            [1.0e-46, 5.0e-46],
            [9.0e-46, 13.0e-46],
        ]
    )
    logac = np.log10(coefficient).astype(np.float32)
    nu_grid = np.array([500.0, 1500.0, 2500.0], dtype=np.float32)

    actual = interp_logacia_matrix(
        np.array([300.0], dtype=np.float32), nu_grid, nucia, tcia, logac
    )

    expected = np.log10(np.array([[5.0e-46, 7.0e-46, 9.0e-46]]))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)
    vector = interp_logacia_vector(300.0, nu_grid, nucia, tcia, logac)
    np.testing.assert_allclose(vector, actual[0], rtol=1.0e-6)
    assert np.all(np.isfinite(actual))


def test_interp_logacia_preserves_large_float32_dynamic_range():
    logac = np.array([[-40.0, -100.0], [-41.0, -101.0]], dtype=np.float32)
    nu_grid = np.array([1000.0, 2000.0], dtype=np.float32)
    nucia = np.array([1000.0, 2000.0], dtype=np.float32)
    tcia = np.array([200.0, 400.0], dtype=np.float32)
    actual = interp_logacia_vector(
        np.float32(300.0),
        nu_grid,
        nucia,
        tcia,
        logac,
    )
    expected = np.log10(
        0.5 * 10.0 ** logac[0].astype(float)
        + 0.5 * 10.0 ** logac[1].astype(float)
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)
    assert np.all(np.isfinite(actual))
    gradient = jax.grad(
        lambda temperature: interp_logacia_vector(
            temperature, nu_grid, nucia, tcia, logac
        ).sum()
    )(np.float32(300.0))
    assert np.isfinite(gradient)


def test_interp_logacia_gradient_at_native_temperature_and_nan():
    tcia = np.array([200.0, 300.0, 400.0], dtype=np.float32)
    nucia = np.array([1000.0, 2000.0], dtype=np.float32)
    nu_grid = np.array([1000.0], dtype=np.float32)
    coefficient = np.array([1.0, 4.0, 9.0], dtype=np.float32) * 1.0e-30
    logac = np.log10(np.column_stack((coefficient, coefficient)))

    gradient = jax.grad(
        lambda temperature: interp_logacia_vector(
            temperature, nu_grid, nucia, tcia, logac
        ).sum()
    )(np.float32(300.0))
    expected_gradient = 5.0 / (400.0 * np.log(10.0))
    assert gradient == pytest.approx(expected_gradient, rel=1.0e-5)

    nan_result = interp_logacia_vector(
        np.float32(np.nan), nu_grid, nucia, tcia, logac
    )
    assert np.all(np.isnan(nan_result))


def test_digitize_reproduces_legacy_interpolation():
    tcia = np.array([200.0, 400.0])
    nucia = np.array([1000.0, 2000.0])
    logac = np.log10(
        np.array(
            [
                [1.0e-40, 5.0e-40],
                [9.0e-40, 13.0e-40],
            ]
        )
    )
    nu_grid = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0])

    actual = interp_logacia_vector(
        300.0,
        nu_grid,
        nucia,
        tcia,
        logac,
        wavenumber_interpolation="digitize",
    )
    native_at_temperature = np.array(
        [
            np.interp(300.0, tcia, logac[:, index])
            for index in range(nucia.size)
        ]
    )
    indices = np.clip(np.digitize(nu_grid, nucia), 0, nucia.size - 1)
    np.testing.assert_allclose(actual, native_at_temperature[indices])


def test_opacia_defaults_to_interp_and_accepts_digitize():
    class Cdb:
        nucia = np.array([1000.0, 2000.0])
        tcia = np.array([200.0, 400.0])
        logac = np.log10(
            np.array(
                [
                    [1.0e-40, 5.0e-40],
                    [9.0e-40, 13.0e-40],
                ]
            )
        )

    nu_grid = np.array([1500.0])
    default = OpaCIA(Cdb(), nu_grid)
    legacy = OpaCIA(Cdb(), nu_grid, wavenumber_interpolation="digitize")

    assert default.wavenumber_interpolation == "interp"
    assert legacy.wavenumber_interpolation == "digitize"
    assert not np.allclose(
        default.logacia_vector(300.0),
        legacy.logacia_vector(300.0),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="wavenumber_interpolation"):
        OpaCIA(Cdb(), nu_grid, wavenumber_interpolation="unknown")


if __name__ == "__main__":
    test_interp_logacia_matrix()
    test_interp_logacia_vector()
