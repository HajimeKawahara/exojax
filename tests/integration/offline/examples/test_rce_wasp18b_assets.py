"""Exercise pinned scientific assets without downloading stellar or line data."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def assets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("_rce_wasp18b_assets")


def write_manifest(tmp_path, files, **config):
    entries = {
        name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "source": "synthetic offline fixture"}
        for name, path in files.items()
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "exojax.rce-wasp18b/1",
                                "files": entries, **config}))
    return path


def stellar_fixture(tmp_path, **kwargs):
    path = tmp_path / "stellar.npz"
    np.savez(path, nu_cm1=[1000., 2000., 4000.], surface_flux_per_cm1=[1., 3., 5.])
    return write_manifest(tmp_path, {"star": path}, stellar={
        "format": "npz", "version": "fixture/1", "asset": "star",
        "flux_unit": "erg s-1 cm-2 (cm-1)-1", **kwargs,
    })


def test_npz_stellar_units_coverage_and_provenance(assets, tmp_path):
    path = stellar_fixture(tmp_path, metadata={"teff_K": 6400.})
    manifest = assets.load_manifest(path)
    spectrum = assets.load_stellar_spectrum(manifest, [1000., 4000.])
    np.testing.assert_array_equal(spectrum.at([1000., 1500., 4000.]), [1., 2., 5.])
    assert spectrum.metadata["teff_K"] == 6400.
    assert spectrum.metadata["version"] == "fixture/1"
    assert manifest.paths["star"] == tmp_path / "stellar.npz"
    assert manifest.sha256["__manifest__"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not cover"):
        spectrum.at([999.])
    with pytest.raises(ValueError, match="does not cover"):
        assets.load_stellar_spectrum(manifest, [4001.])


@pytest.mark.parametrize("updates,message", [
    ({"flux_unit": "MJy"}, "flux_unit"),
    ({"version": ""}, "version"),
    ({"format": "blackbody"}, "Unsupported stellar format"),
])
def test_stellar_requires_explicit_units_and_version(assets, tmp_path, updates, message):
    manifest = assets.load_manifest(stellar_fixture(tmp_path, **updates))
    with pytest.raises(ValueError, match=message):
        assets.load_stellar_spectrum(manifest)


def test_tampered_asset_is_rejected_before_parsing(assets, tmp_path):
    path = stellar_fixture(tmp_path)
    (tmp_path / "stellar.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        assets.load_manifest(path)


def test_missing_asset_has_actionable_error(assets, tmp_path):
    path = stellar_fixture(tmp_path)
    (tmp_path / "stellar.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Missing asset 'star'"):
        assets.load_manifest(path)


@pytest.mark.parametrize("field,value,message", [
    ("schema_version", "unknown", "schema_version"),
    ("sha256", "unrecorded", "SHA256 hex digest"),
    ("source", "", "explicit source"),
])
def test_manifest_requires_schema_and_provenance(assets, tmp_path, field, value, message):
    path = stellar_fixture(tmp_path)
    config = json.loads(path.read_text())
    if field == "schema_version":
        config[field] = value
    else:
        config["files"]["star"][field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        assets.load_manifest(path)


@pytest.mark.parametrize("nu,flux,message", [
    ([1., 1.], [1., 2.], "unique"),
    ([1., np.nan], [1., 2.], "finite and positive"),
    ([1., 2.], [1., -2.], "nonnegative"),
    ([1., 2.], [1., np.inf], "nonnegative"),
    ([1., 2.], [0., 0.], "nonzero"),
    ([1., 2.], [1.], "matching 1D"),
])
def test_invalid_stellar_spectrum_is_rejected(assets, tmp_path, nu, flux, message):
    path = stellar_fixture(tmp_path)
    config = json.loads(path.read_text())
    star_path = tmp_path / "stellar.npz"
    np.savez(star_path, nu_cm1=nu, surface_flux_per_cm1=flux)
    path = write_manifest(tmp_path, {"star": star_path}, stellar=config["stellar"])
    with pytest.raises(ValueError, match=message):
        assets.load_stellar_spectrum(assets.load_manifest(path))


@pytest.mark.parametrize("wavelength_headers", [
    {"BUNIT": "Angstrom"}, {"UNIT": "Angstrom"},
    {"BUNIT": "Angstrom", "UNIT": "Angstrom"},
])
def test_phoenix_conversion_preserves_surface_flux_and_headers(assets, tmp_path,
                                                              wavelength_headers):
    from astropy import units as u
    from astropy.constants import c
    from astropy.io import fits

    wave = np.array([10000., 20000., 40000.])
    flux = np.array([1.e12, 2.e12, 3.e12])
    wave_path, flux_path = tmp_path / "wave.fits", tmp_path / "flux.fits"
    fits.writeto(wave_path, wave, fits.Header(wavelength_headers))
    header = fits.Header({"BUNIT": "erg/s/cm^2/cm", "PHXTEFF": 6400.,
                          "PHXLOGG": 4., "PHXVER": "16.01.00"})
    fits.writeto(flux_path, flux, header)
    path = write_manifest(tmp_path, {"wave": wave_path, "flux": flux_path}, stellar={
        "format": "phoenix-fits", "version": "fixture-PHOENIX/1",
        "wavelength_asset": "wave", "flux_asset": "flux",
        "wavelength_unit": "angstrom", "flux_unit": "erg s-1 cm-2 cm-1",
    })
    spectrum = assets.load_stellar_spectrum(assets.load_manifest(path))
    per_hz = (flux * u.erg / u.s / u.cm**3).to_value(
        u.erg / u.s / u.cm**2 / u.Hz,
        equivalencies=u.spectral_density(wave * u.AA),
    )
    expected = per_hz * c.to_value(u.cm / u.s)
    np.testing.assert_allclose(spectrum.nu_cm1, [2500., 5000., 10000.])
    np.testing.assert_allclose(spectrum.surface_flux_per_cm1, expected[::-1])
    assert spectrum.metadata["teff_K"] == 6400.
    assert spectrum.metadata["logg"] == 4.
    assert spectrum.metadata["phoenix_version"] == "16.01.00"
    assert spectrum.metadata["header"]["PHXVER"] == "16.01.00"
    config = json.loads(path.read_text())
    config["stellar"]["wavelength_unit"] = "nm"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="UNIT disagrees"):
        assets.load_stellar_spectrum(assets.load_manifest(path))


def test_phoenix_rejects_conflicting_wavelength_unit_headers(assets, tmp_path):
    from astropy.io import fits

    wave_path, flux_path = tmp_path / "wave.fits", tmp_path / "flux.fits"
    fits.writeto(wave_path, np.array([10000., 20000.]),
                 fits.Header({"BUNIT": "Angstrom", "UNIT": "nm"}))
    fits.writeto(flux_path, np.ones(2), fits.Header({"BUNIT": "erg/s/cm^2/cm"}))
    path = write_manifest(tmp_path, {"wave": wave_path, "flux": flux_path}, stellar={
        "format": "phoenix-fits", "version": "fixture-PHOENIX/1",
        "wavelength_asset": "wave", "flux_asset": "flux",
        "wavelength_unit": "angstrom", "flux_unit": "erg s-1 cm-2 cm-1",
    })
    with pytest.raises(ValueError, match="wavelength UNIT disagrees"):
        assets.load_stellar_spectrum(assets.load_manifest(path))


def test_existing_ckd_and_diffgrid_loaders_are_reused(assets, tmp_path):
    import jax.numpy as jnp
    from exojax.opacity import OpaCKD, OpaDiffgrid, saveopa

    class AnalyticTeacher:
        method = "analytic-test"
        ready = True
        nu_grid = np.linspace(1000., 1002., 9)

        def xsmatrix(self, temperature, pressure):
            return jnp.broadcast_to(
                1.e-23 * jnp.exp(-700. / jnp.asarray(temperature))[:, None],
                (len(pressure), len(self.nu_grid)),
            )

    teacher = AnalyticTeacher()
    ck_path, lbl_path = tmp_path / "ck.npz", tmp_path / "lbl.npz"
    ckd = OpaCKD(teacher, Ng=2, band_width=1., band_spacing="linear")
    ckd.precompute_tables(jnp.array([700., 1500.]), jnp.array([0.1, 1.]), to_path=ck_path)
    lbl = OpaDiffgrid(teacher, np.array([700., 1500.]), np.array([0.1, 1.]))
    saveopa(lbl, str(lbl_path), format="npz")
    path = write_manifest(tmp_path, {
        "ck": ck_path, "lbl": lbl_path, "lbl_metadata": tmp_path / "lbl_metadata.json",
    }, opacity={
        "ckd": {"H2O": {"asset": "ck", "format": "npz"}},
        "lbl": {"H2O": {"asset": "lbl", "format": "diffgrid", "metadata_asset": "lbl_metadata"}},
    })
    manifest = assets.load_manifest(path)
    restored_ckd = assets.load_ckd_opacities(manifest)["H2O"]
    restored_lbl = assets.load_lbl_opacities(manifest)["H2O"]
    assert restored_ckd.base_opa is None
    np.testing.assert_array_equal(restored_ckd.ckd_info.log_kggrid, ckd.ckd_info.log_kggrid)
    np.testing.assert_allclose(restored_lbl.xsmatrix(jnp.array([900., 1100.]), jnp.array([0.1, 1.])),
                               lbl.xsmatrix(jnp.array([900., 1100.]), jnp.array([0.1, 1.])))
    with pytest.raises(ValueError, match="not declared"):
        assets.load_opacity(manifest, {"asset": "lbl", "format": "diffgrid"})
    (tmp_path / "lbl_metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch.*lbl_metadata"):
        assets.load_manifest(path)


def exomolop_fixture(tmp_path, *, edges=(100., 200., 500., 1400.),
                     k_unit="cm^2/molecule", p_unit="bar"):
    import h5py

    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as table:
        table["mol_name"], table["mol_mass"] = [b"H2O"], [18.]
        table["bin_centers"] = [140., 300., 800.]
        if edges is not None:
            table["bin_edges"] = edges
        table["samples"], table["weights"] = [0.25, 0.75], [0.5, 0.5]
        table["t"], table["p"] = [1500., 3000.], [0.01, 10.]
        table["kcoeff"] = np.full((2, 2, 3, 2), 1.e-23)
        table["kcoeff"].attrs["units"] = k_unit
        table["p"].attrs["units"] = p_unit
    return write_manifest(tmp_path, {"native": path})


def test_exomolop_preserves_native_unequal_band_widths(assets, tmp_path):
    manifest = assets.load_manifest(exomolop_fixture(tmp_path))
    opa = assets.load_opacity(manifest, {"asset": "native", "format": "exomolop"})
    expected = np.array([[100., 200.], [200., 500.], [500., 1400.]])
    np.testing.assert_array_equal(opa.band_edges, expected)
    np.testing.assert_array_equal(opa.ckd_info.band_edges, expected)
    np.testing.assert_array_equal(np.diff(opa.band_edges, axis=1).ravel(), [100., 300., 900.])
    assert opa.band_width == 300.
    np.testing.assert_allclose(np.exp(opa.ckd_info.log_kggrid), 1.e-23, rtol=2.e-6)


@pytest.mark.parametrize("updates,message", [
    ({"k_unit": "m^2/molecule"}, "kcoeff units"),
    ({"p_unit": "Pa"}, "p units"),
    ({"edges": None}, "requires native bin_edges"),
    ({"edges": (100., 500., 1400.)}, "bin_edges must increase"),
    ({"edges": (100., 500., 200., 1400.)}, "bin_edges must increase"),
    ({"edges": (100., 200., np.nan, 1400.)}, "bin_edges must increase"),
    ({"edges": (100., 200., 900., 1400.)}, "bin_edges must increase"),
])
def test_exomolop_rejects_unusable_native_metadata(assets, tmp_path, updates, message):
    manifest = assets.load_manifest(exomolop_fixture(tmp_path, **updates))
    with pytest.raises(ValueError, match=message):
        assets.load_opacity(manifest, {"asset": "native", "format": "exomolop"})
