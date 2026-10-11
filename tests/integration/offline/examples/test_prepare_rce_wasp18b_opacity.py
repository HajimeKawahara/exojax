"""Verify preparation, resumability and P7 archives without network access."""

import importlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_opacity")


def make_args(preparation, tmp_path, extra=()):
    source = tmp_path / "existing.txt"
    source.write_text("pinned asset")
    manifest = dict(
        schema_version=preparation.SCHEMA_VERSION,
        files={"existing": preparation._file_entry(source, "offline fixture")},
        column={"pressure_boundaries_bar": [0.001, 0.01, 0.1]},
        opacity=dict(ckd={"C1O1": {}, "H3N1": {}}, lbl={}, mapping={"C1O1": {"isotope_fraction": 1.0}, "H3N1": {}}),
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(manifest))
    return preparation.parser_with_defaults().parse_args([
        "--database-root", str(tmp_path / "database"), "--inventory", str(inventory),
        "--output-dir", str(tmp_path / "products"), "--species", "C1O1",
        "--nu-min", "4300", "--nu-max", "4400", "--resolved-nu-min", "4320", "--resolved-nu-max", "4380",
        "--ckd-resolution", "200", "--samples-per-band", "8", "--bands-per-chunk", "2",
        "--temperature-nodes", "2", "--ng", "2", "--wing-padding", "1", *extra,
    ])


class AnalyticTeacher:
    method = "analytic-fixture"
    ready = True
    molmass = 28.0

    def __init__(self, nu):
        self.nu_grid = nu

    def xsmatrix(self, temperature, pressure):
        value = 1.e-23 * jnp.exp(-1000. / temperature) * (1. + pressure)
        return jnp.broadcast_to(value[:, None], (len(pressure), len(self.nu_grid)))


def test_dry_run_grid_boundaries_and_memory(preparation, tmp_path, capsys):
    args = make_args(preparation, tmp_path)
    plan = preparation.make_plan(args)
    assert plan["number_of_chunks"] == 3
    assert plan["resolved_points"] < plan["broadband_teacher_points"]
    previous_edge, previous_nu = None, None
    for index in range(plan["number_of_chunks"]):
        padded, nu, crop, edges = preparation.chunk_grid(plan, index)
        assert padded.size % 2 == 0
        assert padded[0] < nu[0] < nu[-1] < padded[-1]
        np.testing.assert_allclose(padded[crop], nu, rtol=1.e-14)
        if previous_edge is not None:
            assert previous_edge == edges[0, 0]
            assert previous_nu == nu[0]
        previous_edge, previous_nu = edges[-1, 1], nu[-1]
    assert previous_edge == 4400.
    original = plan["grid_fingerprint"]
    args.temperature_max = 4000.
    assert preparation.make_plan(args)["grid_fingerprint"] != original
    assert not Path(plan["output_dir"]).exists()


def test_matched_tables_merge_manifest_and_resume(preparation, tmp_path):
    args = make_args(preparation, tmp_path)
    plan = preparation.make_plan(args)
    spec = plan["settings"]
    records = []
    with jax.experimental.enable_x64():
        for index in range(plan["number_of_chunks"]):
            padded, nu, crop, edges = preparation.chunk_grid(plan, index)
            directory = Path(plan["output_dir"]) / "C1O1" / f"chunk_{index:05d}"
            directory.mkdir(parents=True)
            provenance = dict(profile_kernel=None, sources={}, no_lines_reason=None)
            teacher = preparation.CroppedTeacher(AnalyticTeacher(padded), nu, crop, provenance)
            preparation.build_tables(teacher, edges, np.array(spec["temperature_grid_K"]),
                                     np.array(spec["pressure_grid_bar"]), spec["ng"], directory, provenance,
                                     (spec["resolved_nu_min"], spec["resolved_nu_max"]))
            files = {name: preparation._file_entry(directory / name, "offline analytic fixture")
                     for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json") if (directory / name).exists()}
            record = dict(grid_fingerprint=plan["grid_fingerprint"], mdb_path=plan["mdb_path"],
                          files=files, provenance=provenance)
            preparation._write_json(directory / "complete.json", record)
            records.append(record)
        assert preparation.completed_chunks(plan) == records
        preparation.merge_species(plan, records)
        manifest_path = preparation.assemble_manifest(plan, ["C1O1"])
        assets = importlib.import_module("_rce_wasp18b_assets")
        manifest = assets.load_manifest(manifest_path)
        assert set(manifest.config["opacity"]["ckd"]) == {"C1O1"}
        assert "H3N1" in manifest.config["opacity"]["omitted_absorbers"]
        ckd = assets.load_ckd_opacities(manifest)["C1O1"]
        lbl = assets.load_lbl_opacities(manifest)["C1O1"]
        assert lbl.nu_grid.size == plan["resolved_points"]
        assert np.all(np.diff(lbl.nu_grid) > 0)
        assert ckd.band_edges[0, 0] == 4300.
        assert ckd.band_edges[-1, -1] == 4400.
        np.testing.assert_array_equal(ckd.band_edges[1:, 0], ckd.band_edges[:-1, -1])
        pressure = jnp.array(spec["pressure_grid_bar"])
        temperature = jnp.full(pressure.shape, spec["temperature_max"])
        expected = AnalyticTeacher(lbl.nu_grid).xsmatrix(temperature, pressure)
        np.testing.assert_allclose(lbl.xsmatrix(temperature), expected, rtol=1.e-12)
        np.testing.assert_allclose(np.asarray(ckd.xstensor_ckd(temperature, pressure))[:, 0, 0], expected[:, 0], rtol=1.e-12)
    corrupt = Path(records[0]["files"]["ckd.npz"]["path"])
    corrupt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        preparation.completed_chunks(plan)


def test_changed_source_and_grid_are_rejected(preparation, tmp_path):
    args = make_args(preparation, tmp_path)
    plan = preparation.make_plan(args)
    directory = Path(plan["output_dir"]) / "C1O1" / "chunk_00000"
    directory.mkdir(parents=True)
    preparation._write_json(directory / "complete.json", dict(grid_fingerprint="different", mdb_path=plan["mdb_path"]))
    with pytest.raises(ValueError, match="different settings"):
        preparation.completed_chunks(plan)
    source = tmp_path / "source.def"
    source.write_text("original")
    record = preparation._file_entry(source, "fixture")
    record["stat"] = [source.stat().st_size, source.stat().st_mtime_ns]
    source.write_text("tampered source")
    with pytest.raises(ValueError, match="Changed source"):
        preparation._check_sources({"source": record})


def test_real_bundled_co_chunk(preparation, tmp_path, monkeypatch):
    """Exercise MdbExomol activation, PreMODIT derivatives and archive loading."""
    from exojax.test.data import get_testdata_filename
    from exojax.database.exomol.api import MdbExomol

    source = Path(get_testdata_filename("CO"))
    database = tmp_path / "CO"
    shutil.copytree(source, database)
    mdb_path = database / "12C-16O" / "SAMPLE"
    # The bundled fixture has both broadening files; disable network probing.
    def offline_database(*args, **kwargs):
        kwargs["broadf_download"] = False
        return MdbExomol(*args, **kwargs)
    monkeypatch.setattr("exojax.database.exomol.api.MdbExomol", offline_database)
    args = make_args(preparation, tmp_path, [
        "--mdb-path", str(mdb_path), "--nu-min", "4329", "--nu-max", "4335",
        "--resolved-nu-min", "4329", "--resolved-nu-max", "4335", "--wing-padding", "1",
        "--samples-per-band", "64", "--ckd-resolution", "1000",
    ])
    plan = preparation.make_plan(args)
    with jax.experimental.enable_x64():
        preparation.prepare_chunk(plan, 0)
        records = preparation.completed_chunks(plan)
        assert records[0]["provenance"]["number_of_lines"] > 0
        assert records[0]["provenance"]["sources"]
        preparation.merge_species(plan, records)
        assets = importlib.import_module("_rce_wasp18b_assets")
        manifest = assets.load_manifest(preparation.assemble_manifest(plan, ["C1O1"]))
        ckd = assets.load_ckd_opacities(manifest)["C1O1"]
        lbl = assets.load_lbl_opacities(manifest)["C1O1"]
        temperatures = jnp.array([2000., 3000.])
        assert np.all(np.isfinite(ckd.xstensor_ckd(temperatures, jnp.array(plan["settings"]["pressure_grid_bar"])) ))
        assert np.all(np.isfinite(lbl.xsmatrix(temperatures)))


def test_teacher_batching_preserves_values_and_derivatives(preparation):
    class BoundedTeacher(AnalyticTeacher):
        def xsmatrix(self, temperature, pressure):
            assert len(temperature) <= 4
            return super().xsmatrix(temperature, pressure)
    nu = np.linspace(4300., 4400., 32)
    teacher = preparation.CroppedTeacher(BoundedTeacher(nu), nu[4:20], slice(4, 20), {})
    temperature, pressure = jnp.linspace(2000., 3000., 9), jnp.linspace(.01, 1., 9)
    expected = AnalyticTeacher(nu[4:20]).xsmatrix(temperature, pressure)
    value, derivative = jax.jvp(lambda t: teacher.xsmatrix(t, pressure), (temperature,), (jnp.ones_like(temperature),))
    np.testing.assert_allclose(value, expected, rtol=1.e-6)
    np.testing.assert_allclose(derivative, expected * (1000. / temperature**2)[:, None], rtol=1.e-6)
