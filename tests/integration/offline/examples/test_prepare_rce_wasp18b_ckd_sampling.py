"""Sampling sidecars reuse one teacher while preserving frozen completions."""

import argparse
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tests.integration.offline.examples.test_prepare_rce_wasp18b_ckd_refinement as fixtures


@pytest.fixture
def sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    old_arguments = fixtures.make_args
    monkeypatch.setattr(fixtures, "make_args", lambda base, path, extra:
                        old_arguments(base, path, [*extra, "--samples-per-band", "512"]))
    yield importlib.import_module("prepare_rce_wasp18b_ckd_sampling")
    jax.config.update("jax_enable_x64", previous)


def setup_plan(sampling, tmp_path):
    native = fixtures.source_fixture(sampling.refinement, tmp_path)
    source = Path(native["source_preparation_path"])
    args = argparse.Namespace(base_preparation=source, source_preparation=source,
        refinement_dir=Path(native["output_dir"]), output_dir=tmp_path / "sampling", species=native["species"])
    return sampling.make_plan(args), args


def install_loader(sampling, plan, monkeypatch):
    native, teachers = plan["refinement_plan"], []

    def loader(original, index):
        _, nu, _, edges = sampling.base.chunk_grid(original, index)
        _, source = sampling.refinement._source_chunk(native, index)
        teacher = fixtures.Teacher(nu, source["provenance"])
        teachers.append(teacher)
        spec = original["settings"]
        directory = Path(original["output_dir"]) / original["species"] / f"chunk_{index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        sampling.base.build_tables(teacher, edges, np.asarray(spec["temperature_grid_K"]),
            np.asarray(spec["pressure_grid_bar"]), spec["ng"], directory, teacher.provenance)
        sampling.base._write_json(directory / "complete.json", {"must_not_escape": True})

    monkeypatch.setattr(sampling.base, "prepare_chunk", loader)
    return teachers


def test_one_teacher_four_outputs_and_existing_refinement_bytes(sampling, tmp_path, monkeypatch):
    from exojax.opacity import OpaCKD

    plan, _ = setup_plan(sampling, tmp_path)
    teachers = install_loader(sampling, plan, monkeypatch)
    native = plan["refinement_plan"]
    sampling.refinement.prepare_chunk(native, 0)
    record_path = sampling.refinement._chunk_path(native, 0)
    old = json.loads(record_path.read_text())
    preserved = {record_path: record_path.read_bytes()}
    preserved.update({Path(entry["path"]): Path(entry["path"]).read_bytes() for entry in old["files"].values()})
    # Existing refinements remain byte-identical, including their completion.
    sampling.prepare_chunk(plan, 0)
    assert teachers[-1].calls == 1
    for path, content in preserved.items():
        assert path.read_bytes() == content
    # Missing refinements and both sampling tables share one expensive call.
    sampling.prepare_chunk(plan, 1)
    assert teachers[-1].calls == 1
    assert sampling.refinement.completed_chunks(native)[1] is not None
    sampled = sampling.completed_chunk(plan, 1)
    assert sampled["source_grid_fingerprint"] == native["source_plan"]["grid_fingerprint"]
    _, nu, _, edges = sampling.base.chunk_grid(native["source_plan"], 1)
    for name, variant in plan["variants"].items():
        subset = nu[::variant["stride"]]
        np.testing.assert_array_equal(subset[::variant["intervals_per_band"]], np.r_[edges[:, 0], edges[-1, 1]])
        expected_teacher = fixtures.Teacher(subset, {})
        expected = OpaCKD(expected_teacher, Ng=16, band_width=1.)
        expected.band_edges = jnp.asarray(edges)
        expected.nu_bands = jnp.sqrt(expected.band_edges[:, 0] * expected.band_edges[:, 1])
        spec = native["source_plan"]["settings"]
        expected.precompute_tables(np.sort(spec["temperature_grid_K"]), np.array(spec["pressure_grid_bar"]))
        actual = OpaCKD.load_only().load_tables(sampled["files"][name]["path"])
        np.testing.assert_array_equal(actual.ckd_info.log_kggrid, expected.ckd_info.log_kggrid)
    calls = len(teachers)
    sampling.prepare_chunk(plan, 1)
    assert len(teachers) == calls
    for args in ((nu + 1., edges, spec["temperature_grid_K"], spec["pressure_grid_bar"]),
                 (nu, edges + .1, spec["temperature_grid_K"], spec["pressure_grid_bar"]),
                 (nu, edges, np.array(spec["temperature_grid_K"]) + 1., spec["pressure_grid_bar"]),
                 (nu, edges, spec["temperature_grid_K"], np.array(spec["pressure_grid_bar"]) * 2.)):
        teacher = fixtures.Teacher(args[0], {})
        with pytest.raises(ValueError, match="exact original"):
            sampling._check_grid(plan, 1, teacher, *args[1:])


def test_prefill_only_exact_nodes_preserves_native_and_rejects_changes(sampling, tmp_path):
    plan, _ = setup_plan(sampling, tmp_path)
    assert sampling.prefill_chunk(plan, 0)["status"] == "ineligible"
    result = sampling.prefill_chunk(plan, 1)
    assert result["status"] == "prefilled"
    record = sampling.completed_chunk(plan, 1)
    assert not record["execution"]["native_teacher_recomputed"]
    assert record["execution"]["original_ckd_reconstruction"]["max_absolute_log_k_difference"] <= 1e-12
    assert not Path(plan["refinement_plan"]["output_dir"]).exists()
    path = sampling._chunk_path(plan, 1)
    before = path.read_bytes()
    assert sampling.prefill_chunk(plan, 1)["status"] == "skipped_existing"
    assert path.read_bytes() == before
    changed = dict(plan, fingerprint="changed implementation")
    with pytest.raises(ValueError, match="different settings or implementation"):
        sampling.completed_chunk(changed, 1)
    source = Path(record["inputs"]["ckd.npz"]["path"])
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        sampling.completed_chunk(plan, 1)


def test_parent_dispatch_resume_merge_and_manifest_keep_original_lbl(sampling, tmp_path, monkeypatch):
    plan, args = setup_plan(sampling, tmp_path)
    teachers = install_loader(sampling, plan, monkeypatch)
    commands = []

    def child(command, check):
        assert check and Path(command[1]).resolve() == Path(sampling.__file__).resolve()
        commands.append(command)
        assert sampling.main(command[2:]) == 0

    monkeypatch.setattr(sampling.subprocess, "run", child)
    argv = ["--base-preparation", str(args.base_preparation), "--source-preparation", str(args.source_preparation),
            "--refinement-dir", str(args.refinement_dir), "--output-dir", str(args.output_dir), "--species", args.species]
    assert sampling.main(argv + ["--dry-run"]) == 0
    assert not args.output_dir.exists()
    assert sampling.main(argv) == 0
    assert len(commands) == plan["number_of_chunks"]
    assert all(teacher.calls == 1 for teacher in teachers)
    completed_bytes = {path: path.read_bytes() for path in args.refinement_dir.rglob("complete.json")}
    calls = len(commands)
    assert sampling.main(argv) == 0 and len(commands) == calls
    for path, value in completed_bytes.items():
        assert path.read_bytes() == value
    baseline_path = sampling.base.assemble_manifest(plan["refinement_plan"]["source_plan"], [args.species])
    baseline = json.loads(baseline_path.read_text())
    for name in plan["variants"]:
        target = tmp_path / f"manifest_{name}.json"
        sampling.assemble_manifest(baseline_path, args.output_dir, name, target)
        actual = json.loads(target.read_text())
        assert actual["opacity"]["lbl"] == baseline["opacity"]["lbl"]
        for key in ("C1O1_lbl", "C1O1_lbl_meta", "C1O1_preparation"):
            assert actual["files"][key] == baseline["files"][key]
        assert actual["preparation"]["grid_fingerprint"] == baseline["preparation"]["grid_fingerprint"]
        assert actual["ckd_sampling"]["variant"]["fingerprint"] != actual["preparation"]["grid_fingerprint"]
        with pytest.raises(ValueError, match="new manifest"):
            sampling.assemble_manifest(baseline_path, args.output_dir, name, target)
    record = sampling.completed_chunk(plan, 1)
    artifact = Path(record["files"]["sampling128"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        sampling.completed_chunk(plan, 1)
