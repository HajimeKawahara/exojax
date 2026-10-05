"""Validate candidate loading and calibrated-pixel binning without private data."""

import hashlib
import importlib
import json
from pathlib import Path

from astropy import units as u
from astropy.table import Table
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def observation_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("_rce_observation")


def write_product(directory, table, recipe, wave, mask):
    """Generate a small candidate and its own integrity manifest."""
    table.write(directory / "emission_candidate.ecsv", format="ascii.ecsv", overwrite=True)
    (directory / "recipe.json").write_text(json.dumps(recipe))
    np.savez(directory / "diagnostics.npz", input_wavelength_um=wave, static_pixel_mask=mask)
    summary = {
        "schema_version": recipe["schema_version"], "target": "synthetic",
        "status": "candidate", "spectral_order": 1, "input_flux_unit": "MJy",
        "n_bins": len(table), "n_accepted": int(np.sum(table["accepted"])),
        "n_static_pixels": int(np.sum(mask)),
        "outputs": {
            name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                   "size_bytes": (directory / name).stat().st_size}
            for name in ("emission_candidate.ecsv", "recipe.json", "diagnostics.npz")
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory / "emission_candidate.ecsv"


@pytest.fixture
def product(tmp_path):
    table = Table({
        "wavelength_um": [1.25, 2.5, 4.0, 12.0],
        "wavelength_lower_um": [1.0, 2.0, 4.0, 8.0],
        "wavelength_upper_um": [2.0, 4.0, 8.0, 16.0],
        "n_pixels": [2, 2, 1, 0],
        "fp_fs": [0.001, 0.002, np.nan, np.nan],
        "uncertainty": [2e-5, 4e-5, np.nan, np.nan],
        "accepted": [True, True, False, False],
        "mask_reason": ["", "", "insufficient_selected_pixels", "insufficient_selected_pixels"],
        "red_noise_beta": [1.1, 1.3, np.nan, np.nan],
        "n_used": [100, 100, 0, 0],
        "fp_fs_ppm": [1000., 2000., np.nan, np.nan],
        "uncertainty_ppm": [20., 40., np.nan, np.nan],
    })
    table.meta = {"status": "candidate", "target": "synthetic", "spectral_order": 1,
                  "event_type": "secondary_eclipse"}
    recipe = {"schema_version": "exospecdb.wasp18b-soss-emission-candidate/0.1.0",
              "target": "synthetic", "spectral_order": 1,
              "wavelength_range_um": [1., 16.], "resolving_power": 1.,
              "minimum_bin_pixels": 2, "radius_ratio": 0.1}
    wave = np.array([1., 1.5, 2., 2.5, 3., 4., 8., 16., 20.])
    mask = np.array([True, True, True, False, True, True, False, False, False])
    return tmp_path, table, recipe, wave, mask


def test_loader_preserves_masks_rows_errors_and_provenance(observation_module, product):
    observation = observation_module.load_emission_candidate(write_product(*product))
    np.testing.assert_array_equal(observation.row_indices, [0, 1])
    np.testing.assert_array_equal(observation.rejected_row_indices, [2, 3])
    assert list(observation.table["mask_reason"][2:]) == ["insufficient_selected_pixels"] * 2
    assert np.all(np.isnan(observation.table["fp_fs"][2:]))
    np.testing.assert_array_equal(observation.fp_fs, [0.001, 0.002])
    # Values already contain beta; loading never inflates them a second time.
    np.testing.assert_array_equal(observation.uncertainty, [2e-5, 4e-5])
    np.testing.assert_array_equal(observation.W_all.sum(axis=1), [2, 2, 1, 0])
    np.testing.assert_array_equal(observation.W, [
        [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0]])
    assert set(observation.sha256) == {"summary.json", "emission_candidate.ecsv",
                                       "recipe.json", "diagnostics.npz"}


def test_final_upper_edge_included_and_internal_edges_not_duplicated(observation_module, product):
    directory, table, recipe, wave, mask = product
    mask[7] = True  # Exactly the upper endpoint, in the otherwise empty final bin.
    table["n_pixels"][-1] = 1
    table["wavelength_um"][-1] = 16.
    observation = observation_module.load_emission_candidate(write_product(*product))
    np.testing.assert_array_equal(observation.W_all.sum(axis=0), mask)
    assert observation.W_all[-1, 7]
    assert not observation.W_all[0, 2]
    assert observation.W_all[1, 2]


@pytest.mark.parametrize("filename", ["emission_candidate.ecsv", "recipe.json", "diagnostics.npz"])
def test_changed_output_is_rejected_before_parsing(observation_module, product, filename):
    path = write_product(*product)
    with (path.parent / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        observation_module.load_emission_candidate(path)


@pytest.mark.parametrize("column,value,message", [
    ("fp_fs", np.nan, "nonfinite fp_fs"),
    ("uncertainty", 0., "uncertainty must be positive"),
    ("uncertainty", -1., "uncertainty must be positive"),
    ("uncertainty", np.inf, "nonfinite uncertainty"),
    ("fp_fs_ppm", 2., "fp_fs_ppm disagrees"),
    ("uncertainty_ppm", 2., "uncertainty_ppm disagrees"),
    ("red_noise_beta", 0.9, "red_noise_beta must"),
    ("n_pixels", 10, "pixel counts"),
    ("wavelength_um", 1.3, "mean wavelengths"),
    ("wavelength_lower_um", 0.9, "bin edges"),
    ("mask_reason", "rejected", "accepted and mask_reason"),
])
def test_invalid_accepted_rows_fail(observation_module, product, column, value, message):
    product[1][column][0] = value
    with pytest.raises(ValueError, match=message):
        observation_module.load_emission_candidate(write_product(*product))


def test_units_and_schema_are_explicit(observation_module, product):
    table = product[1]
    table["wavelength_um"].unit = u.um
    table["fp_fs"].unit = u.dimensionless_unscaled
    observation_module.load_emission_candidate(write_product(*product))
    table["wavelength_um"].unit = u.nm
    with pytest.raises(ValueError, match="unexpected unit"):
        observation_module.load_emission_candidate(write_product(*product))
    table["wavelength_um"].unit = u.um
    product[2]["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="schema_version"):
        observation_module.load_emission_candidate(write_product(*product))


def test_invalid_pixel_mask_and_wavelengths(observation_module, product):
    directory, table, recipe, wave, mask = product
    with pytest.raises(ValueError, match="static_pixel_mask"):
        observation_module.load_emission_candidate(
            write_product(directory, table, recipe, wave, mask.astype(int)))
    wave[0] = np.nan
    with pytest.raises(ValueError, match="input_wavelength_um"):
        observation_module.load_emission_candidate(write_product(*product))


def test_ratio_preserves_constant_and_uses_stellar_flux_weights(observation_module, product):
    observation = observation_module.load_emission_candidate(write_product(*product))
    star = np.arange(1., 10.)
    ratio = observation_module.pixel_flux_ratio
    np.testing.assert_allclose(ratio(observation.W, 0.2 * star, star, 0.01), 0.002)
    planet = np.arange(9., 0., -1.)
    expected = 0.01 * np.array([(9 + 8) / (1 + 2), (7 + 5) / (3 + 5)])
    result = ratio(observation.W, planet, star, 0.01)
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    mean_ratios = 0.01 * (observation.W @ (planet / star)) / observation.W.sum(axis=1)
    assert not np.allclose(result, mean_ratios)
    # A common Fnu unit scale (e.g. Jy to MJy) cancels before ppm display conversion.
    np.testing.assert_allclose(ratio(observation.W, planet * 1e-6, star * 1e-6, 0.01), result)
    np.testing.assert_allclose(result * 1e6, expected * 1e6)


@pytest.mark.parametrize("length_unit", ["um", "nm", "angstrom", "cm", "m"])
def test_flambda_conversion_matches_independent_spectral_density_equivalency(observation_module, length_unit):
    wave = np.array([0.9, 1.5, 2.8])
    flux = np.array([1., 2., 4.])
    expected = (flux * u.erg / u.s / u.cm**2 / u.Unit(length_unit)).to_value(
        u.erg / u.s / u.cm**2 / u.Hz, equivalencies=u.spectral_density(wave * u.um))
    converted = observation_module.flambda_to_fnu(wave, flux, per_wavelength=length_unit)
    np.testing.assert_allclose(converted, expected, rtol=1e-14)


def test_flambda_conversion_precedes_binning_and_supports_jit_grad(observation_module, product):
    observation = observation_module.load_emission_candidate(write_product(*product))
    wave = observation.input_wavelength_um
    planet = jnp.arange(9., 0., -1.)
    star = jnp.arange(1., 10.)
    convert = observation_module.flambda_to_fnu
    ratio = observation_module.pixel_flux_ratio

    def model(scale):
        return ratio(observation.W, convert(wave, scale * planet), convert(wave, star), 0.01)

    expected = 0.01 * (observation.W @ (np.asarray(planet) * wave**2)) / (
        observation.W @ (np.asarray(star) * wave**2))
    np.testing.assert_allclose(jax.jit(model)(1.), expected, rtol=1e-14)
    assert not np.allclose(expected, ratio(observation.W, planet, star, 0.01))
    value, tangent = jax.jit(lambda x: jax.jvp(model, (x,), (1.,)))(2.)
    np.testing.assert_allclose(value, 2 * expected, rtol=1e-14)
    np.testing.assert_allclose(tangent, expected, rtol=1e-14)
    derivative = jax.jit(jax.grad(lambda x: jnp.sum(model(x))))(2.)
    np.testing.assert_allclose(derivative, expected.sum(), rtol=1e-14)


def test_operator_rejects_ambiguous_shapes_and_units(observation_module):
    with pytest.raises(ValueError, match="pixel spectra"):
        observation_module.pixel_flux_ratio(np.ones((2, 3)), np.ones(2), np.ones(2), 0.01)
    with pytest.raises(ValueError, match="unsupported"):
        observation_module.flambda_to_fnu(np.ones(3), np.ones(3), per_wavelength="Hz")
