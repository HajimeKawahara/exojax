"""Saved-node refinement must prove eligibility and preserve frozen provenance."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tests.integration.offline.examples.test_prepare_rce_wasp18b_ckd_refinement import source_fixture


@pytest.fixture
def prefill(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("prefill_rce_wasp18b_ckd_refinement")
    jax.config.update("jax_enable_x64", previous)


def test_saved_teacher_requires_complete_unclipped_exact_nodes(prefill):
    nu, temperature, pressure = np.array([4300., 4310., 4320.]), np.array([3000., 2000.]), np.array([.01, 1.])
    logs = np.arange(12, dtype=float).reshape(2, 2, 3) * .1 - 50.
    info = SimpleNamespace(temperature_grid=temperature, pressure_grid=pressure,
        log_cross_section_grid=logs, log_cross_section_floor=-80.)
    opa = SimpleNamespace(nu_grid=nu, diffgrid_info=info)
    teacher = prefill.SavedNodeTeacher(opa, nu, temperature, pressure, {})
    actual = teacher.xsmatrix(np.array([2000., 3000., 2000.]), np.array([1., .01, .01]))
    np.testing.assert_allclose(actual, np.exp(logs[[1, 0, 0], [1, 0, 1]]), rtol=1e-15)
    with pytest.raises(ValueError, match="exact original"):
        teacher.xsmatrix(np.array([2500.]), np.array([.01]))
    with pytest.raises(ValueError, match="exact original"):
        teacher.xsmatrix(np.array([2000.]), np.array([.1]))
    with pytest.raises(prefill.IneligibleChunk, match="full chunk core"):
        prefill.SavedNodeTeacher(opa, nu[1:], temperature, pressure, {})
    info.log_cross_section_grid[0, 0, 0] = -80.
    with pytest.raises(prefill.IneligibleChunk, match="Floor-clipped"):
        prefill.SavedNodeTeacher(opa, nu, temperature, pressure, {})


def test_prefill_is_compatible_with_frozen_resume_and_pins_actual_inputs(prefill, tmp_path):
    plan = source_fixture(prefill.refinement, tmp_path)
    base, refinement = prefill.base, prefill.refinement
    result = prefill.prefill_chunk(plan, 1, dry_run=True)
    assert result["status"] == "eligible"
    assert not Path(plan["output_dir"]).exists()
    result = prefill.prefill_chunk(plan, 1)
    assert result["status"] == "prefilled"
    record_path = Path(result["completion"])
    original_bytes = record_path.read_bytes()
    record = refinement.completed_chunks(plan)[1]
    execution = record["provenance"]["refinement_execution"]
    assert execution["method"] == "saved_diffgrid_nodes" and not execution["native_teacher_recomputed"]
    assert execution["original_ckd_reconstruction"]["max_absolute_log_k_difference"] <= 1e-12
    assert set(execution["inputs"]) == {"diffgrid.npz", "diffgrid_metadata.json", "ckd.npz", "prefill_driver"}
    for entry in execution["inputs"].values():
        assert record["provenance"]["sources"][entry["path"]]["sha256"] == entry["sha256"]
    for entry in record["files"].values():
        with np.load(entry["path"], allow_pickle=False) as archive:
            metadata = json.loads(archive["meta"].tobytes())["base_fingerprint"]
        assert metadata["teacher"]["refinement_execution"] == execution
    assert prefill.prefill_chunk(plan, 1)["status"] == "skipped_existing"
    assert record_path.read_bytes() == original_bytes
    # The unchanged native resume validator must notice changed reused inputs.
    source_ckd = Path(execution["inputs"]["ckd.npz"]["path"])
    source_ckd.write_bytes(source_ckd.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed source"):
        refinement.completed_chunks(plan)


def test_partial_core_and_existing_outputs_are_left_to_native_job(prefill, tmp_path):
    plan = source_fixture(prefill.refinement, tmp_path)
    result = prefill.prefill_chunk(plan, 0)
    assert result["status"] == "ineligible" and "full chunk core" in result["reason"]
    assert not Path(plan["output_dir"]).exists()
    partial = Path(plan["output_dir"]) / "ng32" / plan["species"] / "chunk_00001"
    partial.mkdir(parents=True)
    sentinel = partial / "ckd.tmp.npz"
    sentinel.write_bytes(b"native work in progress")
    assert prefill.prefill_chunk(plan, 1)["status"] == "skipped_existing"
    assert sentinel.read_bytes() == b"native work in progress"


def test_original_ckd_mismatch_and_changed_input_are_rejected(prefill, tmp_path):
    from exojax.opacity import OpaDiffgrid

    plan = source_fixture(prefill.refinement, tmp_path)
    _, record = prefill.refinement._source_chunk(plan, 1)
    _, nu, _, edges = prefill.base.chunk_grid(plan["source_plan"], 1)
    spec = plan["source_plan"]["settings"]
    temperature, pressure = np.array(spec["temperature_grid_K"]), np.array(spec["pressure_grid_bar"])
    opa = OpaDiffgrid.from_saved_opa(record["files"]["diffgrid.npz"]["path"])
    teacher = prefill.SavedNodeTeacher(opa, nu, temperature, pressure, {})
    teacher.log_values = teacher.log_values + 1e-8
    with pytest.raises(prefill.IneligibleChunk, match="exceeds log"):
        prefill.validate_original_ckd(teacher, edges, temperature, pressure,
                                      record["files"]["ckd.npz"]["path"], spec["ng"])
    path = Path(record["files"]["diffgrid.npz"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        prefill.prefill_chunk(plan, 1)
    assert not Path(plan["output_dir"]).exists()
