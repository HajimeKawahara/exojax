"""Check Planck units, untouched native spectra, and pinned preparation outputs."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_stellar")


def native_planck(preparation, amplitude=1.7):
    from _rce_wasp18b_assets import StellarSpectrum

    nu = np.geomspace(1e4 / 5.499975, 1e4 / 0.05, 12000)
    flux = amplitude * preparation.planck_surface_flux(nu, 6400.)
    return StellarSpectrum(nu, flux, {"teff_K": 6400.})


def write_phoenix_manifest(preparation, tmp_path):
    from astropy.io import fits
    from _rce_wasp18b_assets import PHOENIX_FLUX_UNIT, SCHEMA_VERSION

    native = native_planck(preparation)
    wave = (1 / native.nu_cm1)[::-1]
    flux = (native.surface_flux_per_cm1 / (1 / native.nu_cm1)**2)[::-1]
    wave_path, flux_path = tmp_path / "wave.fits", tmp_path / "flux.fits"
    fits.writeto(wave_path, wave * 1e8)
    fits.writeto(flux_path, flux, header=fits.Header({"PHXTEFF": 6400.}))
    config = {
        "schema_version": SCHEMA_VERSION,
        "files": {key: {"path": path.name, "sha256": preparation._sha256(path),
                         "source": "offline Planck fixture"}
                  for key, path in (("wave", wave_path), ("flux", flux_path))},
        "stellar": {"format": "phoenix-fits", "version": "test-native",
                    "wavelength_asset": "wave", "flux_asset": "flux",
                    "wavelength_unit": "angstrom", "flux_unit": PHOENIX_FLUX_UNIT},
        "unrelated": {"preserve": True},
        "missing_requirements": ["Stellar surface spectrum covering the complete broadband grid.",
                                 "Other audit."],
    }
    path = tmp_path / "prepared_manifest.json"
    path.write_text(json.dumps(config))
    return path


def test_planck_units_and_integrated_flux_recovery(preparation):
    from astropy import units as u
    from astropy.modeling.models import BlackBody

    nu = np.geomspace(0.01, 1e6, 100000)
    flux = preparation.planck_surface_flux(nu, 6400.)
    # Astropy returns radiance per Hz; c converts it to radiance per cm-1.
    expected = (np.pi * BlackBody(temperature=6400 * u.K)(nu * u.cm**-1)
                * (2.99792458e10 * u.cm / u.s)).to_value(u.erg / u.s / u.cm / u.sr)
    np.testing.assert_allclose(flux, expected, rtol=2e-12, atol=0.)
    assert np.trapz(flux, nu) == pytest.approx(preparation.SIGMA_CGS * 6400**4, rel=1e-8)
    native = native_planck(preparation)
    extended_nu, extended_flux, metadata = preparation.extend_spectrum(native)
    assert metadata["amplitude"] == pytest.approx(1.7, rel=1e-10)
    np.testing.assert_allclose(extended_flux,
                               1.7 * preparation.planck_surface_flux(extended_nu, 6400.), rtol=1e-10)


def test_native_samples_soss_and_join_are_preserved(preparation):
    from _rce_wasp18b_assets import StellarSpectrum

    native = native_planck(preparation)
    # A resolved feature at the endpoint stresses the exterior-only join.
    native.surface_flux_per_cm1[:4] *= [0.6, 0.9, 0.8, 0.95]
    nu, flux, metadata = preparation.extend_spectrum(native)
    n = native.nu_cm1.size
    np.testing.assert_array_equal(nu[-n:], native.nu_cm1)
    np.testing.assert_array_equal(flux[-n:], native.surface_flux_per_cm1)
    extended = StellarSpectrum(nu, flux, metadata)
    soss_nu = np.linspace(1e4 / 2.85, 1e4 / 0.85, 1000)
    np.testing.assert_array_equal(extended.at(soss_nu), native.at(soss_nu))
    extended.validate_coverage([100., 1e5])
    assert np.all(np.diff(nu) > 0) and np.all(flux >= 0)
    assert flux[-n - 1] / flux[-n] == pytest.approx(1., rel=0.004)
    assert metadata["fit_window_um"][1] == pytest.approx(5.499975)
    assert metadata["tail_integral_erg_s_cm2"] > 0
    assert metadata["total_integral_erg_s_cm2"] == pytest.approx(np.trapz(flux, nu))


@pytest.mark.parametrize("options,message", [
    ({"temperature": -1}, "temperature"),
    ({"fit_window": (5.5, 4.5)}, "fit_window"),
    ({"fit_window": (4.5, 6.)}, "within the native"),
    ({"fit_window": (np.nan, 5.5)}, "fit_window"),
    ({"max_wavelength": 90}, "at least 100"),
    ({"blend_width": 0}, "blend width"),
    ({"tail_samples": 2}, "tail_samples"),
])
def test_invalid_choices_are_rejected(preparation, options, message):
    with pytest.raises(ValueError, match=message):
        preparation.extend_spectrum(native_planck(preparation), **options)


def test_manifest_pins_native_and_generated_files_and_dry_run(preparation, tmp_path):
    from _rce_wasp18b_assets import load_manifest, load_stellar_spectrum

    source = write_phoenix_manifest(preparation, tmp_path)
    original = source.read_bytes()
    output = tmp_path / "separate" / "stellar"
    arguments = ["--manifest", str(source), "--output-dir", str(output)]
    preparation.main(arguments + ["--dry-run"])
    assert not output.exists()
    preparation.main(arguments)
    assert source.read_bytes() == original
    manifest = load_manifest(output / "stellar_manifest.json")
    result = load_stellar_spectrum(manifest, [100., 1e5])
    native = load_stellar_spectrum(load_manifest(source))
    np.testing.assert_array_equal(result.nu_cm1[-native.nu_cm1.size:], native.nu_cm1)
    np.testing.assert_array_equal(result.surface_flux_per_cm1[-native.nu_cm1.size:],
                                  native.surface_flux_per_cm1)
    assert manifest.config["unrelated"] == {"preserve": True}
    assert manifest.config["missing_requirements"] == [
        "Other audit.", "Stellar tail fitting-window/temperature and broadband endpoint sensitivity."]
    metadata = json.loads(manifest.paths["stellar_preparation"].read_text())
    assert metadata["native_sha256"] == {key: manifest.sha256[key] for key in ("wave", "flux")}
    assert metadata["input_manifest"]["sha256"] == preparation._sha256(source)
    assert metadata["temperature_source"] == "PHOENIX PHXTEFF"
    assert set(manifest.paths) == {"wave", "flux", "stellar_extended", "stellar_preparation"}
    with np.load(manifest.paths["stellar_extended"], allow_pickle=False) as data:
        assert json.loads(str(data["metadata_json"])) == metadata
    with pytest.raises(SystemExit):
        preparation.main(arguments)
    manifest.paths["stellar_preparation"].write_text("tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_manifest(output / "stellar_manifest.json")


def test_corrupt_native_pin_prevents_preparation(preparation, tmp_path):
    source = write_phoenix_manifest(preparation, tmp_path)
    (tmp_path / "flux.fits").write_bytes(b"tampered")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        preparation.main(["--manifest", str(source), "--output-dir", str(output)])
    assert not output.exists()
