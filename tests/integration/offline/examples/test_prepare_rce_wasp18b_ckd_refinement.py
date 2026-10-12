"""Verify paired CKD refinements preserve the exact original teacher inputs."""

import argparse
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_prepare_rce_wasp18b_opacity import make_args


@pytest.fixture
def refinement(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("prepare_rce_wasp18b_ckd_refinement")
    jax.config.update("jax_enable_x64", previous)


class Teacher:
    method, ready, molmass = "analytic-refinement-fixture", True, 28.

    def __init__(self, nu, provenance):
        self.nu_grid, self.provenance, self.calls = nu, provenance, 0

    def meta(self):
        return self.provenance

    def xsmatrix(self, temperature, pressure):
        self.calls += 1
        profile = 1 + 50 * np.exp(-((self.nu_grid - 4350.) / 3.)**2)
        return 1e-23 * jnp.exp(-1000. / temperature[:, None]) * (1 + pressure[:, None]) * profile


def source_fixture(refinement, tmp_path):
    base = refinement.base
    args = make_args(base, tmp_path, ["--ng", "16"])
    source_plan = base.make_plan(args)
    spec, records = source_plan["settings"], []
    for index in range(source_plan["number_of_chunks"]):
        _, nu, _, edges = base.chunk_grid(source_plan, index)
        directory = Path(source_plan["output_dir"]) / source_plan["species"] / f"chunk_{index:05d}"
        directory.mkdir(parents=True)
        provenance = dict(sources={}, profile_kernel=spec["profile_kernel"], no_lines_reason=None,
                          number_of_lines=4, wing_padding_cm1=spec["wing_padding"])
        base.build_tables(Teacher(nu, provenance), edges, np.array(spec["temperature_grid_K"]),
                          np.array(spec["pressure_grid_bar"]), spec["ng"], directory, provenance,
                          (spec["resolved_nu_min"], spec["resolved_nu_max"]))
        files = {name: base._file_entry(directory / name, "Synthetic original teacher")
                 for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json") if (directory / name).exists()}
        record = dict(grid_fingerprint=source_plan["grid_fingerprint"], mdb_path=source_plan["mdb_path"],
                      files=files, provenance=provenance)
        base._write_json(directory / "complete.json", record)
        records.append(record)
    base.merge_species(source_plan, records)
    source_path = Path(source_plan["output_dir"]) / source_plan["species"] / "complete.json"
    return refinement.make_plan(argparse.Namespace(
        base_preparation=source_path, source_preparation=source_path,
        output_dir=tmp_path / "refinement", species=source_plan["species"],
    ))


def test_cached_teacher_rejects_changed_coordinates_and_inputs(refinement):
    teacher = Teacher(np.array([4300., 4400.]), {})
    cached = refinement.CachedTeacher(teacher)
    temperature, pressure = jnp.array([2200., 3000.]), jnp.array([.01, 1.])
    first = cached.xsmatrix(temperature, pressure)
    assert cached.xsmatrix(temperature, pressure) is first and teacher.calls == 1
    for t, p in ((temperature + 1., pressure), (temperature, pressure * 2.)):
        with pytest.raises(ValueError, match="same T/P"):
            cached.xsmatrix(t, p)
    teacher.nu_grid = np.array([4300., 4401.])
    with pytest.raises(ValueError, match="wavenumbers changed"):
        cached.xsmatrix(temperature, pressure)


def test_pair_build_resume_merge_and_original_lbl_reference(refinement, tmp_path, monkeypatch):
    from exojax.opacity import OpaCKD

    plan = source_fixture(refinement, tmp_path)
    base, teachers = refinement.base, []
    original_source = Path(plan["source_preparation_path"])
    source_hash = base._sha256(original_source)
    source_record = json.loads(original_source.read_text())

    def fake_loader(original, index):
        _, nu, _, edges = base.chunk_grid(original, index)
        _, source = refinement._source_chunk(plan, index)
        directory = Path(original["output_dir"]) / original["species"] / f"chunk_{index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        teacher = Teacher(nu, source["provenance"])
        teachers.append(teacher)
        spec = original["settings"]
        base.build_tables(teacher, edges, np.asarray(spec["temperature_grid_K"]),
                          np.asarray(spec["pressure_grid_bar"]), spec["ng"], directory, teacher.provenance)
        # This original-grid completion must not escape the refinement adapter.
        base._write_json(directory / "complete.json", {"unexpected_original_completion": True})

    monkeypatch.setattr(base, "prepare_chunk", fake_loader)
    for index in range(plan["number_of_chunks"]):
        refinement.prepare_chunk(plan, index)
    assert all(teacher.calls == 1 for teacher in teachers)
    records = refinement.completed_chunks(plan)
    assert all(record["fingerprint"] == plan["fingerprint"] for record in records)
    outputs = refinement.merge_species(plan, records)
    assert len(outputs) == 2 and base._sha256(original_source) == source_hash
    assert not list(Path(plan["output_dir"]).rglob("diffgrid*.npz"))
    tables = {}
    for output in outputs:
        record = json.loads(Path(output).read_text())
        variant = record["variant"]
        assert variant["grid_fingerprint"] != plan["source_plan"]["grid_fingerprint"]
        assert record["original_resolved_files"]["diffgrid.npz"] == source_record["files"]["diffgrid.npz"]
        table = OpaCKD.load_only().load_tables(record["files"]["ckd.npz"]["path"])
        assert table.ckd_info.log_kggrid.shape[2:] == (
            variant["ng"], plan["source_plan"]["settings"]["number_of_bands"] * variant["band_split"])
        np.testing.assert_array_equal(table.band_edges[1:, 0], table.band_edges[:-1, 1])
        tables[variant["band_split"]] = table
    np.testing.assert_array_equal(tables[1].band_edges[:, 0], tables[2].band_edges[::2, 0])
    np.testing.assert_array_equal(tables[1].band_edges[:, 1], tables[2].band_edges[1::2, 1])
    baseline_path = base.assemble_manifest(plan["source_plan"], [plan["species"]])
    baseline = json.loads(baseline_path.read_text())
    for variant in plan["variants"]:
        target = tmp_path / f"manifest_{variant}.json"
        assert refinement.assemble_manifest(baseline_path, plan["output_dir"], variant, target) == target
        assembled = json.loads(target.read_text())
        assert assembled["opacity"]["lbl"] == baseline["opacity"]["lbl"]
        for key in ("C1O1_lbl", "C1O1_lbl_meta", "C1O1_preparation"):
            assert assembled["files"][key] == baseline["files"][key]
        assert assembled["files"]["C1O1_ckd"]["sha256"] != baseline["files"]["C1O1_ckd"]["sha256"]
        with pytest.raises(ValueError, match="new manifest"):
            refinement.assemble_manifest(baseline_path, plan["output_dir"], variant, target)
    mismatch = json.loads(baseline_path.read_text())
    mismatch["opacity"]["ckd"]["unexpected"] = {}
    wrong_baseline = tmp_path / "wrong_species.json"
    wrong_baseline.write_text(json.dumps(mismatch))
    with pytest.raises(ValueError, match="exactly match"):
        refinement.assemble_manifest(wrong_baseline, plan["output_dir"], "ng32", tmp_path / "must_not_exist.json")
    completion = Path(outputs[0])
    original_record = completion.read_text()
    corrupted = json.loads(original_record)
    corrupted["variant"] = plan["variants"]["bands2"]
    completion.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="provenance"):
        refinement.assemble_manifest(baseline_path, plan["output_dir"], "ng32", tmp_path / "must_not_exist.json")
    completion.write_text(original_record)
    # Independent direct calls to the ordinary builder must reproduce every bin.
    for index, teacher in enumerate(teachers):
        _, nu, _, edges = base.chunk_grid(plan["source_plan"], index)
        for name, grid in refinement.refined_edges(plan, index, nu, edges).items():
            variant = plan["variants"][name]
            expected = OpaCKD(teacher, Ng=variant["ng"], band_width=1., band_spacing="log")
            expected.band_edges = jnp.asarray(grid)
            expected.nu_bands = jnp.sqrt(expected.band_edges[:, 0] * expected.band_edges[:, 1])
            expected.precompute_tables(tables[1].ckd_info.T_grid, tables[1].ckd_info.P_grid)
            actual = OpaCKD.load_only().load_tables(records[index]["files"][name]["path"])
            np.testing.assert_array_equal(actual.ckd_info.log_kggrid, expected.ckd_info.log_kggrid)
    artifact = Path(records[0]["files"]["ng32"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        refinement.completed_chunks(plan)


def test_dry_run_and_grid_guard(refinement, tmp_path, capsys):
    plan = source_fixture(refinement, tmp_path)
    path = Path(plan["source_preparation_path"])
    output = tmp_path / "dryrun"
    assert refinement.main(["--base-preparation", str(path), "--source-preparation", str(path),
                            "--species", "C1O1", "--output-dir", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert plan["variants"]["ng32"]["grid_fingerprint"] != plan["variants"]["bands2"]["grid_fingerprint"]
    _, nu, _, edges = refinement.base.chunk_grid(plan["source_plan"], 0)
    with pytest.raises(ValueError, match="pinned original chunk"):
        refinement.refined_edges(plan, 0, nu + 1., edges)
    altered = json.loads(path.read_text())
    altered["plan"]["settings"]["samples_per_band"] += 2
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="fingerprint"):
        refinement.make_plan(argparse.Namespace(base_preparation=path, source_preparation=path,
                                               output_dir=output, species="C1O1"))
