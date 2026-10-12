"""Verify paired native additions without full-grid preparation or new physics."""

from copy import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules
from tests.integration.offline.examples.test_prepare_rce_wasp18b_supplemental import arguments


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("rce_wasp18b_molecular_omission_audit")
    jax.clear_caches()
    jax.config.update("jax_enable_x64", previous)


class Teacher:
    nu_grid = np.linspace(1., 3., 9)

    def __init__(self):
        self.calls = []

    def xsmatrix(self, temperature, pressure):
        self.calls.append((np.asarray(temperature), np.asarray(pressure)))
        return (temperature[:, None] * pressure[:, None]
                * jnp.asarray(self.nu_grid)[None, :])


def test_teacher_uses_actual_pairs_and_inclusive_native_masks(audit):
    teacher = Teacher()
    temperature, pressure = np.array([1900., 2900., 3900.]), np.array([.01, 1., 10.])
    edges = np.array([[1., 2.], [2., 3.]])
    arrays, info = audit.fixed_teacher_values(teacher, temperature, pressure, edges, 4, (1.5, 2.5), 1e-35)
    assert len(teacher.calls) == 1
    np.testing.assert_array_equal(teacher.calls[0][0], temperature)
    np.testing.assert_array_equal(teacher.calls[0][1], pressure)
    assert info["evaluated_layer_pairs"] == 3
    assert info["band_sample_counts"] == [5, 5]
    assert arrays["logk"].shape == (3, 4, 2)
    g, _ = audit.gauss_legendre_grid(4)
    expected = temperature[:, None] * pressure[:, None] * teacher.nu_grid[:5]
    np.testing.assert_allclose(arrays["logk"][..., 0], audit.compute_ckd_from_xsmatrix(expected, g), rtol=0., atol=0.)
    np.testing.assert_array_equal(arrays["resolved_nu"], teacher.nu_grid[2:7])
    np.testing.assert_allclose(arrays["resolved_xs"], temperature[:, None]*pressure[:, None]*teacher.nu_grid[2:7])


@pytest.mark.parametrize("temperature,pressure", [([1500., np.nan], [1., 2.]), ([1500., 2000.], [1.]), ([1500.], [0.])])
def test_invalid_pairs_are_rejected_before_native_evaluation(audit, temperature, pressure):
    teacher = Teacher()
    with pytest.raises(ValueError, match="aligned finite positive"):
        audit.fixed_teacher_values(teacher, temperature, pressure, [[1., 3.]], 4, (1., 3.), 1e-35)
    assert teacher.calls == []


def test_resolved_floor_and_zero_line_ckd_are_explicit(audit):
    teacher = Teacher()
    teacher.xsmatrix = lambda t, p: jnp.zeros((len(t), len(teacher.nu_grid)))
    arrays, info = audit.fixed_teacher_values(teacher, [2500.], [1.], [[1., 3.]], 4, (1., 3.), 1e-35)
    np.testing.assert_array_equal(arrays["resolved_xs"], np.full((1, 9), 1e-35))
    assert info["resolved_values_below_floor"] == 9
    assert np.all(np.isfinite(arrays["logk"]))


def test_frozen_intercept_restores_functions_after_failure(audit, monkeypatch):
    build, write = audit.base.build_tables, audit.base._write_json
    def failing(*args):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(audit.supplemental, "prepare_chunk", failing)
    with pytest.raises(RuntimeError, match="fixture failure"):
        audit.native_chunk({}, 0, [2500.], [1.], 1e-35)
    assert audit.base.build_tables is build
    assert audit.base._write_json is write


def test_native_chunk_matches_small_frozen_tables_without_rewriting_them(audit, tmp_path, monkeypatch):
    from exojax.database.exomol.api import MdbExomol
    from exojax.test.data import get_testdata_filename

    # CO under the existing OH dispatch label is solely an offline loader fixture.
    database = tmp_path / "CO"
    shutil.copytree(Path(get_testdata_filename("CO")), database)
    def offline_database(*args, **kwargs):
        kwargs["broadf_download"] = False
        return MdbExomol(*args, **kwargs)
    monkeypatch.setattr("exojax.database.exomol.api.MdbExomol", offline_database)
    argv = arguments(audit.supplemental, tmp_path, ["--mdb-path", str(database / "12C-16O/SAMPLE")])
    plan = audit.supplemental.make_plan(audit.supplemental.parser_with_defaults().parse_args(argv))
    audit.supplemental.prepare_chunk(plan, 0)
    original = Path(plan["output_dir"]) / "H1O1/chunk_00000"
    pinned = {p.name: audit._sha256(p) for p in original.iterdir()}
    pressure = np.asarray(plan["settings"]["pressure_grid_bar"])
    temperature = np.asarray(plan["settings"]["temperature_grid_K"])
    assert len(temperature) == len(pressure) == 2
    expected = audit.OpaDiffgrid.from_saved_opa(str(original / "diffgrid.npz")).xsmatrix(jnp.asarray(temperature))
    with np.load(original / "ckd.npz") as data:
        reference = data["log_kggrid"][[1, 0], [0, 1]]
    plan = dict(plan, output_dir=str(tmp_path / "actual_pairs"))
    saved = audit.native_chunk(plan, 0, temperature, pressure, 1e-35)
    np.testing.assert_allclose(saved["arrays"]["logk"], reference, rtol=0., atol=3e-13)
    np.testing.assert_allclose(saved["arrays"]["resolved_xs"], expected, rtol=3e-12, atol=1e-35)
    assert saved["details"]["evaluated_layer_pairs"] == 2
    assert saved["provenance"]["number_of_lines"] > 0
    assert not list((tmp_path / "actual_pairs").rglob("*.npz"))
    assert not list((tmp_path / "actual_pairs").rglob("complete.json"))
    assert pinned == {p.name: audit._sha256(p) for p in original.iterdir()}


def test_checkpoint_reuse_rejects_changed_identity_and_bytes(audit, tmp_path, monkeypatch):
    calls = []
    def native(*args):
        calls.append(args)
        return dict(arrays={"logk": np.ones((2, 4, 1))}, details={}, provenance={"sources": {}})
    monkeypatch.setattr(audit, "native_chunk", native)
    arrays, record = audit.cached_native_chunk(tmp_path, "pin", {}, 0, [2500.], [1.], 1e-35)
    again, resumed = audit.cached_native_chunk(tmp_path, "pin", {}, 0, [2500.], [1.], 1e-35)
    assert len(calls) == 1 and record == resumed
    np.testing.assert_array_equal(arrays["logk"], again["logk"])
    with pytest.raises(ValueError, match="Changed omission"):
        audit.cached_native_chunk(tmp_path, "other", {}, 0, [2500.], [1.], 1e-35)
    (tmp_path / "chunk_00000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed omission"):
        audit.cached_native_chunk(tmp_path, "pin", {}, 0, [2500.], [1.], 1e-35)


def test_additions_keep_rorr_order_and_zero_abundance_identity(audit):
    _, weights = audit.gauss_legendre_grid(4)
    base = jnp.asarray(np.arange(48.).reshape(2, 2, 4, 3)/7.)
    extra = jnp.asarray(np.arange(24.).reshape(1, 2, 4, 3)/11.)
    continuum = jnp.ones((2, 3))*.17
    amounts = jnp.array([[2., 3.]])
    expected = audit.mix_ckd_rorr(jnp.concatenate((base, extra*amounts[..., None, None])), weights)+continuum[:, None]
    actual = audit.mixed_depth(base, extra, amounts, continuum, weights)
    np.testing.assert_array_equal(actual, expected)
    zero = audit.mixed_depth(base, extra, amounts*0., continuum, weights)
    baseline = audit.mix_ckd_rorr(base, weights)+continuum[:, None]
    np.testing.assert_allclose(zero, baseline, rtol=5e-15, atol=1e-14)


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("scattering", [False, True])
def test_depth_override_reuses_production_flux_and_chunk_seam(audit, modules, tmp_path, resolved, scattering):
    manifest, column, chemistry = fixture_manifest(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=scattering)
    temperature, bottom, parameters = jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.69])
    if not resolved:
        broadband = importlib.import_module("_rce_broadband")
        edges = jnp.array([[3500., 3700.], [3700., 4500.]])
        column.band_nodes, column.band_weights = broadband.make_band_quadrature(edges, 4)
        column.band_widths = jnp.diff(edges, axis=1).ravel()
        _, column.g_weights = audit.gauss_legendre_grid(4)
        column.star_bands = broadband.band_stellar_flux(column.stellar.nu_cm1, column.stellar.surface_flux_per_cm1, edges)
        column.ckd = {name: SimpleNamespace(xstensor_ckd=lambda t,p: jnp.broadcast_to(
            jnp.arange(1., 5.)[None, :, None]*1e-24, (2, 4, 2))) for name in column.lbl}
    depth = column.optical_depth(temperature, chemistry, resolved=resolved)
    expected = column.fluxes(temperature, bottom, parameters, chemistry, resolved=resolved)
    actual = audit.flux_at_depth(column, depth, column.nu if resolved else column.band_nodes,
        column.star_resolved if resolved else column.star_bands, temperature, bottom,
        parameters, chemistry, resolved=resolved)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.)
    if resolved:
        _, stream = modules.stream.load_streaming_opacities(manifest)
        grids, spectra = [], {"baseline": []}
        for index, nu in enumerate(stream.grids):
            tables = audit.resolved_tables(stream.chunks[index], nu, column.pressure)
            local = copy(column)
            local.nu, local.star_resolved = jnp.asarray(nu), jnp.asarray(column.stellar.at(nu))
            depth = local.optical_depth(temperature, chemistry, resolved=True, resolved_tables=tables)
            flux = audit.flux_at_depth(local, depth, local.nu, local.star_resolved,
                temperature, bottom, parameters, chemistry, resolved=True).upward[0]
            audit.append_spectrum(grids, spectra, nu, {"baseline": flux})
        np.testing.assert_array_equal(np.concatenate(grids), stream.nu)
        full = column.spectrum(temperature, bottom, parameters, chemistry, resolving_power=700.)
        streamed = stream.prediction(column, np.concatenate(spectra["baseline"]), 700.)
        np.testing.assert_allclose(streamed, full, rtol=2e-12)


def test_nonadjacent_smoke_chunks_do_not_drop_samples(audit):
    grids, spectra = [], {"baseline": []}
    for grid in (np.array([1., 2.]), np.array([3., 4.]), np.array([4., 5.])):
        audit.append_spectrum(grids, spectra, grid, {"baseline": grid**2})
    np.testing.assert_array_equal(np.concatenate(grids), np.arange(1., 6.))
    np.testing.assert_array_equal(np.concatenate(spectra["baseline"]), np.arange(1., 6.)**2)
    with pytest.raises(ValueError, match="ordered"):
        audit.append_spectrum(grids, spectra, np.array([4., 6.]), {"baseline": np.array([16., 36.])})


def test_combined_isomer_population_and_legacy_components_are_counted_once(audit):
    species = ("H1O1", "Fe1H1", "O1V1", "C1H1N1_hcn", "C1H1N1_hnc")
    values = jnp.asarray([[1., 2., 3., 4., 5.], [11., 12., 13., 14., 15.], [21., 22., 23., 24., 25.]])*1e-8
    chemistry = SimpleNamespace(x=values)
    result = audit.component_fractions(audit.SPECIES, chemistry, species)
    for index, name in enumerate(species[:3]):
        np.testing.assert_array_equal(result[name], values[:-1, index])
    np.testing.assert_array_equal(result["HCN_HNC"], values[:-1, 3]+values[:-1, 4])
    with pytest.raises(ValueError, match="double-count"):
        audit.component_fractions(["HCN_HNC", "HCN_HNC"], chemistry, species)


def test_hcn_requires_explicit_verified_asset_and_vo_has_exact_mapping(audit, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requires --hcn-hnc-asset"):
        audit.component_database("HCN_HNC", tmp_path)
    database, mapping = audit.component_database("O1V1", tmp_path)
    assert database == tmp_path / "VO/51V-16O/VOMYT"
    assert mapping["chemical_species"] == ("O1V1",)
    marker = tmp_path / "direct_partition.json"
    marker.write_text("fixture metadata")
    monkeypatch.setattr(audit.hcn_partition, "verify_asset", lambda path: {"fixture_verified": True})
    database, mapping = audit.component_database("HCN_HNC", tmp_path, tmp_path)
    assert mapping["chemical_species"] == ("C1H1N1_hcn", "C1H1N1_hnc")
    assert mapping["partition_asset"] == {"fixture_verified": True}
    assert mapping["partition_metadata_sha256"] == audit._sha256(marker)
