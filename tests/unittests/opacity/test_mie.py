"""opacity for mie test
"""
from types import SimpleNamespace

from exojax.test.emulate_pdb import mock_PdbPlouds
from exojax.opacity import OpaMie
from exojax.utils.grids import wavenumber_grid
import numpy as np
import jax.numpy as jnp
import pytest


def test_mieparams_vector_direct_uses_wavelength_nm(monkeypatch):
    wavenumber = np.array([5000.0, 10000.0, 15000.0])
    wavelength_nm = 1.0e7 / wavenumber
    pdb = SimpleNamespace(
        refraction_index_wavenumber=wavenumber,
        refraction_index_wavelength_nm=wavelength_nm,
        refraction_index=np.ones(3, dtype=complex),
        N0=1.0,
    )
    opa = OpaMie(pdb, wavenumber)
    passed_wavelengths = []

    def mock_mie_lognormal(m, wavelength, sigmag, rg, N0, rgrid):
        passed_wavelengths.append(wavelength)
        return np.zeros(7)

    monkeypatch.setattr(
        "exojax.opacity.opacont.mie_lognormal", mock_mie_lognormal
    )
    monkeypatch.setattr(
        "exojax.database.mie.auto_rgrid", lambda rg, sigmag: np.ones(1)
    )

    opa.mieparams_vector_direct(rg=1.0e-5, sigmag=2.0)

    np.testing.assert_allclose(passed_wavelengths, wavelength_nm)


def test_mieparams_vector_direct_reference_cross_sections(monkeypatch):
    wavelength_nm = np.array([550.0])
    wavenumber = 1.0e7 / wavelength_nm
    pdb = SimpleNamespace(
        refraction_index_wavenumber=wavenumber,
        refraction_index_wavelength_nm=wavelength_nm,
        refraction_index=np.array([1.5 + 0.01j]),
        N0=3.0,
    )
    opa = OpaMie(pdb, wavenumber)
    monkeypatch.setattr(
        "exojax.database.mie.auto_rgrid",
        lambda rg, sigmag: np.geomspace(10.0, 1500.0, 128),
    )

    actual = opa.mieparams_vector_direct(rg=1.0e-5, sigmag=1.7)

    # PyMieScatt 1.8.1.1 reference at N0=1, converted from Mm^-1 to cm^2.
    expected = [[1.0426828014059518e-9], [9.869338749723963e-10], [0.6824829448345409]]
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=0.0)


def test_mieparams_matrix_direct_uses_scalar_calls(monkeypatch):
    opa = OpaMie(SimpleNamespace(), np.arange(3))

    def mock_mieparams_vector(rg, sigmag):
        rg = float(rg)
        sigmag = float(sigmag)
        return (
            np.full(3, rg),
            np.full(3, sigmag),
            np.full(3, rg + sigmag),
        )

    monkeypatch.setattr(
        opa, "mieparams_vector_direct", mock_mieparams_vector
    )

    result = opa.mieparams_matrix_direct(
        jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    )

    np.testing.assert_allclose(result[0], [[1.0] * 3, [2.0] * 3])
    np.testing.assert_allclose(result[1], [[3.0] * 3, [4.0] * 3])
    np.testing.assert_allclose(result[2], [[4.0] * 3, [6.0] * 3])


def test_mieparams_matrix_direct_mismatched_layers():
    opa = OpaMie(SimpleNamespace(), np.arange(3))

    with pytest.raises(ValueError, match="same length"):
        opa.mieparams_matrix_direct(np.ones(2), np.ones(3))


def test_mieparams_matrix():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    N = 1000
    nus, wav, res = wavenumber_grid(12050.0, 15950.0, N, xsmode="premodit")
    opa = OpaMie(pdb, nus)
    rg_layer = jnp.array([1.0e-5, 2.0e-5])
    sigmag_layer = jnp.array([2.0, 1.0])
    dtau, w, g = opa.mieparams_matrix(rg_layer, sigmag_layer)

    # shape check
    assert np.all(np.shape(dtau) == np.array([len(rg_layer), N]))
    assert np.all(np.shape(w) == np.array([len(rg_layer), N]))
    assert np.all(np.shape(g) == np.array([len(rg_layer), N]))

    expected_vector = opa.mieparams_vector(rg_layer[0], sigmag_layer[0])
    for actual, expected in zip((dtau, w, g), expected_vector):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual[0], expected, rtol=1.0e-6, atol=0.0)
