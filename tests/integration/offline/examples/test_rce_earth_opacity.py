"""Independent line-profile and table I/O checks without HITRAN downloads."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import voigt_profile


@pytest.fixture
def opacity_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("rce_earth_opacity")


@pytest.fixture
def lines():
    return np.array([
        [1000.0, 1000.4],
        np.log([1.0e-20, 3.0e-21]),
        [0.0, 100.0],
        [0.08, 0.05],
        [0.5, 0.8],
        [1.0, 2.0],
    ])


def test_cutoff_and_lorentz_pedestal_against_scipy(opacity_module, lines):
    nu = np.array([974.9, 975.0, 990.0, 999.99, 1000.0, 1000.4, 1010.0, 1025.4, 1025.5])
    state = np.array([[296.0, 1.0, 1.0, 18.010565]])
    actual = np.asarray(opacity_module._chunk_cross_sections(nu, lines, state))[0]
    sigma = 3.0415595e-7 * np.sqrt(296.0 / state[0, 3]) * lines[0]
    gamma = lines[3] / 1.01325 + 2.6544188e-12 * lines[5]
    delta = nu[None, :] - lines[0, :, None]
    pedestal = gamma / (np.pi * (25.0**2 + gamma**2))
    profiles = np.maximum(voigt_profile(delta, sigma[:, None], gamma[:, None]) - pedestal[:, None], 0.0)
    expected = np.exp(lines[1]) @ np.where(np.abs(delta) <= 25.0, profiles, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=1e-35)
    np.testing.assert_array_equal(actual[[0, -1]], 0.0)


def test_generated_table_loads_through_standard_ckd_api(opacity_module, lines, tmp_path):
    opacity = object.__new__(opacity_module.WaterLineOpacity)
    opacity.mdb = SimpleNamespace(qr_interp=lambda isotope, temperature, reference: 1.0)
    opacity.molmass = 18.010565
    opacity.line_data = lines
    opacity.metadata = {"test_line_count": 2}
    temperatures, pressures = np.array([200.0, 300.0]), np.array([0.1, 1.0])
    bands = np.array([[999.8, 1000.6]])
    path = tmp_path / "water_ckd.npz"
    prepared = opacity_module.prepare_table(
        opacity, {"source": "synthetic test lines"}, path, bands,
        temperatures, pressures, dnu=0.002, ng=16, split_g=False, check_ng=32,
    )
    loaded = opacity_module.OpaCKD.from_saved_tables(str(path))
    np.testing.assert_array_equal(loaded.ckd_info.log_kggrid, prepared.ckd_info.log_kggrid)
    np.testing.assert_allclose(
        loaded.xstensor_ckd(temperatures, pressures),
        prepared.xstensor_ckd(temperatures, pressures),
    )
    higher = opacity_module.OpaCKD.from_saved_tables(str(tmp_path / "water_ckd_ng32.npz"))
    assert higher.ckd_info.log_kggrid.shape == (2, 2, 32, 1)
    assert float(np.sum(higher.ckd_info.weights)) == pytest.approx(1.0)


def test_core_gather_matches_full_exojax_voigt(opacity_module, lines):
    nu = np.linspace(974.9, 1025.5, 2531)
    states = np.array([[160.0, 1.0e-4, 0.5, 18.010565], [340.0, 1.0, 1.2, 18.010565]])
    reference = opacity_module._chunk_cross_sections(nu, lines, states)
    actual = opacity_module._chunk_cross_sections(nu, lines, states, np.arange(-4, 5))
    np.testing.assert_allclose(actual, reference, rtol=1.0e-9, atol=1e-34)


def test_thermal_and_shortwave_bands_are_contiguous(opacity_module):
    bands = opacity_module.spectral_bands()
    np.testing.assert_array_equal(bands[1:, 0], bands[:-1, 1])
    assert bands[0, 0] == 20.0
    assert bands[-1, 1] == 30000.0
    assert np.all(np.diff(bands, axis=1) > 0.0)


def test_spectral_chunks_preserve_profiles(opacity_module, lines):
    opacity = object.__new__(opacity_module.WaterLineOpacity)
    opacity.mdb = SimpleNamespace(qr_interp=lambda isotope, temperature, reference: 1.0)
    opacity.molmass = 18.010565
    opacity.line_data = lines
    nu = np.linspace(974.0, 1026.0, 521)
    states = np.array([[296.0, 1.0, 1.0, opacity.molmass]])
    actual = opacity.cross_sections(nu, [296.0], [1.0])
    expected = opacity_module._chunk_cross_sections(nu, lines, states)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-34)


def test_composite_quadrature_uses_midpoint_cdf(opacity_module, tmp_path):
    samples = np.array([1.0, 2.0, 4.0, 8.0]) * 1.0e-23
    opacity = SimpleNamespace(
        molmass=18.010565, metadata={},
        cross_sections=lambda nu, temperature, pressure: np.tile(samples, (len(temperature), 1)),
    )
    result = opacity_module.prepare_table(
        opacity, {"synthetic": True}, tmp_path / "midpoints.npz",
        np.array([[1000.0, 1004.0]]), np.array([296.0]), np.array([1.0]), dnu=1.0,
    )
    g = np.asarray(result.ckd_info.ggrid)
    expected = np.interp(g, [0.125, 0.375, 0.625, 0.875], np.log(samples))
    np.testing.assert_allclose(result.ckd_info.log_kggrid[0, 0, :, 0], expected, atol=1e-13)
    assert g[-1] > 0.99999
    assert float(np.sum(result.ckd_info.weights[-8:])) == pytest.approx(1.0e-4)


@pytest.mark.parametrize("arguments", [
    ["--strength-cutoff", "nan"],
    ["--strength-cutoff", "inf"],
    ["--strength-cutoff=-1e-28"],
    ["--resolution", "nan"],
    ["--resolution", "inf"],
    ["--resolution", "0"],
    ["--dnu", "nan"],
    ["--dnu", "inf"],
    ["--dnu", "0"],
    ["--nu-min", "nan"],
    ["--nu-max", "inf"],
    ["--nu-max", "39976"],
    ["--nu-min", "30000"],
    ["--ng", "16"],
    ["--check-ng", "16"],
    ["--state-batch", "0"],
])
def test_cli_rejects_invalid_parameters_before_data_access(opacity_module, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["rce_earth_opacity.py", *arguments])
    monkeypatch.setattr(opacity_module, "download_water",
                        lambda *args: pytest.fail("Invalid parameters reached data access."))
    with pytest.raises(SystemExit) as error:
        opacity_module.main()
    assert error.value.code == 2


def test_cli_accepts_unsplit_quadrature_and_zero_cutoff(opacity_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rce_earth_opacity.py", "--no-split-g",
                        "--ng", "16", "--check-ng", "32", "--strength-cutoff", "0",
                        "--nu-max", "39975"])
    monkeypatch.setattr(opacity_module, "download_water", lambda *args: (Path("water.par"), {}))
    monkeypatch.setattr(opacity_module, "WaterLineOpacity", lambda *args: SimpleNamespace(metadata={}))
    calls = []
    monkeypatch.setattr(opacity_module, "prepare_table", lambda *args: calls.append(args))
    opacity_module.main()
    assert len(calls) == 1
