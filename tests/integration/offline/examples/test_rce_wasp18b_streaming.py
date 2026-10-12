"""Compare streamed and full P7 transfer, including an observed chunk seam."""

from copy import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity import OpaDiffgrid, saveopa
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import initialize_gaussian_quadrature


class Diagnostics(NamedTuple):
    valid: object


class Chemistry(NamedTuple):
    x: object
    mmw: object
    number_density_e: object
    number_density_h: object
    diagnostics: Diagnostics


class Teacher:
    method, ready = "analytic-fixture", True

    def __init__(self, nu, scale):
        self.nu_grid, self.scale = nu, scale

    def xsmatrix(self, temperature, pressure):
        feature = 1 + 100 * jnp.exp(-((jnp.asarray(self.nu_grid) - 4000.) / 12.)**2)
        return (self.scale * 1e-24 * jnp.exp(-800. / temperature[:, None]) * feature
                * (1 + .1 * jnp.log(pressure[:, None])**2))


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield SimpleNamespace(stream=importlib.import_module("_rce_wasp18b_streaming"),
                          column=importlib.import_module("_rce_wasp18b"),
                          assets=importlib.import_module("_rce_wasp18b_assets"))
    jax.config.update("jax_enable_x64", previous)


def fixture_manifest(modules, tmp_path, *, gap=False, mismatched_species=False, boundaries=None,
                     temperature=None):
    def entry(path):
        return {"path": str(path), "sha256": modules.assets._sha256(path), "source": "offline fixture"}

    nu = np.linspace(3500., 4500., 65)
    boundaries = np.asarray([0.001, 0.1, 10.] if boundaries is None else boundaries)
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = np.asarray([1500., 2500., 3500., 4500.] if temperature is None else temperature)
    species = ("H2O1", "C1O1")
    files, lbl, full = {}, {}, {}
    for name, scale in zip(species, (1., 1.5)):
        directory = tmp_path / name
        directory.mkdir()
        own_nu = nu.copy()
        if mismatched_species and name == "C1O1":
            own_nu[12] += 0.1
        opa = OpaDiffgrid(Teacher(own_nu, scale), temperature, pressure)
        saveopa(opa, str(directory / "diffgrid.npz"), format="npz",
                extra_meta={"grid_fingerprint": "fixture-grid"})
        full[name] = opa
        chunks = []
        for index, selection in enumerate((slice(None, 33), slice(32, None))):
            child_dir = directory / f"chunk_{index:05d}"
            child_dir.mkdir()
            child = OpaDiffgrid.__new__(OpaDiffgrid)
            child.nu_grid = own_nu[selection].copy()
            if gap and index == 1:
                child.nu_grid[0] += 0.1
            child.method, child.teacher_method, child.teacher_profile_kernel = "diffgrid", "analytic-fixture", None
            values = np.asarray(opa.diffgrid_info.log_cross_section_grid)[..., selection].copy()
            if index == 1:
                # The duplicate seam may differ across original halo calculations.
                # The preceding chunk must win exactly as in the merged archive.
                values[..., 0] += 0.4
            child._apply_diffgrid_info(replace(
                opa.diffgrid_info, log_cross_section_grid=jnp.asarray(values),
                log_cross_section_derivative_grid=opa.diffgrid_info.log_cross_section_derivative_grid[..., selection]))
            child.ready = True
            saveopa(child, str(child_dir / "diffgrid.npz"), format="npz",
                    extra_meta={"grid_fingerprint": "fixture-grid"})
            record = {"grid_fingerprint": "fixture-grid", "mdb_path": "offline-lines",
                      "files": {filename: entry(child_dir / filename)
                                for filename in ("diffgrid.npz", "diffgrid_metadata.json")}}
            child_path = child_dir / "complete.json"
            child_path.write_text(json.dumps(record))
            chunks.append(entry(child_path))
        preparation = directory / "complete.json"
        preparation.write_text(json.dumps({
            "plan": {"grid_fingerprint": "fixture-grid", "mdb_path": "offline-lines"},
            "files": {filename: entry(directory / filename)
                      for filename in ("diffgrid.npz", "diffgrid_metadata.json")},
            "chunk_records": chunks,
        }))
        for key, filename in (("lbl", "diffgrid.npz"), ("lbl_meta", "diffgrid_metadata.json"),
                              ("preparation", "complete.json")):
            files[name + "_" + key] = entry(directory / filename)
        lbl[name] = {"asset": name + "_lbl", "metadata_asset": name + "_lbl_meta", "format": "diffgrid"}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": modules.assets.SCHEMA_VERSION,
                                         "files": files, "opacity": {"lbl": lbl}}))
    manifest = modules.assets.load_manifest(manifest_path)

    # A small physical column reuses the actual optical-depth and transfer paths;
    # its supplied chemistry isolates the storage/streaming change from chemistry.
    column = modules.column.Wasp18bColumn.__new__(modules.column.Wasp18bColumn)
    column.lbl, column.nu = full, jnp.asarray(nu)
    column.boundaries, column.pressure = jnp.asarray(boundaries), jnp.asarray(pressure)
    column.temperature_range = (1500., 4500.)
    column.settings = modules.column.ColumnSettings(19000., 200.)
    column.isotope_fractions = {"H2O1": .997, "C1O1": .986}
    column.chemistry = SimpleNamespace(species=species + ("H2", "He1", "H1", "e1-"))
    column.cia = {pair: SimpleNamespace(nucia=jnp.array([3000., 5000.]),
                                       tcia=jnp.array([1500., 4500.]),
                                       logac=jnp.full((2, 2), coefficient))
                  for pair, coefficient in ((('H2', 'H2'), -46.), (('H2', 'He1'), -47.))}
    column.mus, column.angular_weights = initialize_gaussian_quadrature(8)
    star_nu = np.linspace(3000., 5000., 1025)
    star_flux = np.asarray(piBarr(jnp.array([6400.]), star_nu)[0])
    column.stellar = modules.assets.StellarSpectrum(star_nu, star_flux, {})
    column.star_resolved = jnp.asarray(column.stellar.at(nu))
    column.a_over_rstar, column.area_ratio = 3.5, .01
    column.pixel_nu = jnp.array([3985., 4000., 4015., 4100.])
    column.W = jnp.array([[1., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    size = boundaries.size
    chemistry = Chemistry(jnp.array([[.001, .0005, .8, .1985, 0., 0.]] * size),
                          jnp.full(size, 2.3), jnp.full(size, 1e8), jnp.full(size, 1e10),
                          Diagnostics(jnp.ones(size, dtype=bool)))
    return manifest, column, chemistry


def test_streamed_transfer_and_global_response_match_full(modules, tmp_path):
    manifest, column, chemistry = fixture_manifest(modules, tmp_path)
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    streamed_column = copy(column)
    streamed_column.lbl = metadata
    for name, value in metadata.items():
        assert not hasattr(value.diffgrid_info, "log_cross_section_grid")
        value.check_pressure_grid(column.pressure)
    temperature, bottom = jnp.array([2300., 2700.]), 3000.
    parameters = jnp.array([0., .6, np.log(.5)])
    expected = column.fluxes(temperature, bottom, parameters, chemistry, resolved=True).upward[0]
    actual = plan.top_flux(streamed_column, temperature, bottom, parameters, chemistry)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    # The Gaussian at 4000 cm-1 spans the join and the intentionally different
    # discarded duplicate. It must use the completed global spectrum.
    for resolution in (None, 300., np.array([200., 300., 400., 600.])):
        reference = column.spectrum(temperature, bottom, parameters, chemistry, resolving_power=resolution)
        prediction = plan.prediction(streamed_column, actual, resolving_power=resolution)
        np.testing.assert_allclose(prediction, reference, rtol=2e-12, atol=0)
    assert not np.allclose(plan.prediction(streamed_column, actual),
                           plan.prediction(streamed_column, actual, 300.), rtol=1e-4, atol=0)
    assert plan.provenance["chunk_count"] == 2
    assert plan.provenance["all_chunk_coefficients_verified"]
    assert plan.provenance["input_manifest_sha256"] == manifest.sha256["__manifest__"]
    assert len(plan.provenance["file_sha256"]) == 18
    np.testing.assert_array_equal(streamed_column.nu, column.nu)


@pytest.mark.parametrize("option,message", [
    ({"gap": True}, "share exactly one endpoint"),
    ({"mismatched_species": True}, "identical streaming chunk grids"),
])
def test_invalid_chunk_alignment_is_rejected(modules, tmp_path, option, message):
    manifest, _, _ = fixture_manifest(modules, tmp_path, **option)
    with pytest.raises(ValueError, match=message):
        modules.stream.load_streaming_opacities(manifest)


def test_chunk_modified_after_setup_is_rejected(modules, tmp_path):
    manifest, column, chemistry = fixture_manifest(modules, tmp_path)
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    column.lbl = metadata
    plan.chunks[0]["H2O1"].path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        plan.top_flux(column, jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.69]), chemistry)


def test_child_record_and_merged_units_are_checked(modules, tmp_path):
    manifest, _, _ = fixture_manifest(modules, tmp_path)
    child = tmp_path / "H2O1" / "chunk_00000" / "complete.json"
    original = child.read_bytes()
    child.write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        modules.stream.load_streaming_opacities(manifest)
    child.write_bytes(original)
    sidecar = manifest.paths["H2O1_lbl_meta"]
    meta = json.loads(sidecar.read_text())
    meta["units"]["cross_section"] = "m^2"
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        modules.stream.load_streaming_opacities(manifest)
    with pytest.raises(ValueError, match="units"):
        modules.stream._read_coordinates(manifest.paths["H2O1_lbl"], sidecar)


def test_native_pressure_subset_matches_full_table_slices(modules, tmp_path):
    manifest, column, chemistry = fixture_manifest(
        modules, tmp_path, boundaries=np.geomspace(.0001, 10., 5))
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    indices = np.array([0, 2])
    subset = plan.pressure_subset([0, 2, 3]).pressure_subset([0, 1])
    expected_column = copy(column)
    expected_column.pressure = column.pressure[indices]
    # Compare on identical, explicit boundaries; opacity pressure nodes are
    # exact slices and are never interpolated by this streaming operation.
    expected_column.boundaries = column.boundaries[jnp.array([0, 2, 4])]
    expected_column.lbl = {}
    for name, opacity in column.lbl.items():
        sliced = copy(opacity)
        info = opacity.diffgrid_info
        sliced._apply_diffgrid_info(replace(
            info, pressure_grid=info.pressure_grid[indices],
            log_cross_section_grid=info.log_cross_section_grid[indices],
            log_cross_section_derivative_grid=info.log_cross_section_derivative_grid[indices]))
        expected_column.lbl[name] = sliced
    streamed = copy(expected_column)
    streamed.lbl = {name: opa.pressure_subset(indices) for name, opa in metadata.items()}
    for opacity in streamed.lbl.values():
        opacity.check_pressure_grid(streamed.pressure)
    chemistry = jax.tree.map(lambda value: value[jnp.array([0, 2, 4])], chemistry)
    arguments = (jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, np.log(.5)]), chemistry)
    expected = expected_column.fluxes(*arguments, resolved=True).upward[0]
    actual = subset.top_flux(streamed, *arguments)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    np.testing.assert_allclose(subset.prediction(streamed, actual, 300.),
                               expected_column.spectrum(*arguments, resolving_power=300.),
                               rtol=2e-12, atol=0)
    assert subset.provenance["source_pressure_indices"] == [0, 2]
    assert subset.provenance["all_chunk_coefficients_verified"]
    assert not plan.provenance["all_chunk_coefficients_verified"]
    with pytest.raises(ValueError, match="pressure selection"):
        plan.top_flux(streamed, *arguments)
    for invalid in ([0, 0], [2, 0], [0, 4], [0., 2.], [0]):
        with pytest.raises(ValueError, match="pressure indices"):
            plan.pressure_subset(invalid)
    subset.chunks[0]["H2O1"].path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        subset.top_flux(streamed, *arguments)
