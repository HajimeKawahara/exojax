"""Exercise checkpoint publication in temporary directories without GPU work."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[4] / "results/run_rce_wasp18b_h2o_sampling_backlog.py"
    spec = importlib.util.spec_from_file_location("h2o_backlog_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_plans(worker, tmp_path):
    s = worker.sampling
    source = tmp_path / "original/H2O1/chunk_00000/complete.json"
    source.parent.mkdir(parents=True)
    s.base._write_json(source, dict(grid_fingerprint="teacher", mdb_path="raw",
                                   provenance=dict(sources={})))
    plans = []
    for name in ("a", "b"):
        plans.append(dict(output_dir=str(tmp_path / name / "sampling"), species="H2O1",
            fingerprint="sampling", variants=dict(sampling128={}, sampling256={}),
            refinement_plan=dict(output_dir=str(tmp_path / name / "refinement"),
                species="H2O1", fingerprint="refinement",
                source_preparation_path=str(source.parent.parent / "complete.json"),
                source_plan=dict(grid_fingerprint="teacher", mdb_path="raw"),
                base_preparation=worker.pin(source))))
    a, b = plans
    record_path = s.refinement._chunk_path(a["refinement_plan"], 0)
    record_path.parent.mkdir(parents=True)
    files = {}
    for name in ("ng32", "bands2"):
        path = record_path.parent / (name + ".npz")
        path.write_bytes(("unchanged " + name).encode())
        files[name] = worker.pin(path)
    s.base._write_json(record_path, dict(fingerprint="refinement", species="H2O1",
        source_chunk=worker.pin(source), files=files, provenance=dict(sources={})))
    return a, b


def install_child(worker, b, monkeypatch):
    calls = []

    def child(argv, **kwargs):
        calls.append(argv)
        assert argv[:3] == ["/usr/bin/time", "-v", "-o"]
        assert argv[-4:] == ["--species", "H2O1", "--chunk-index", "0"]
        assert argv[argv.index("--refinement-dir") + 1] == b["refinement_plan"]["output_dir"]
        assert argv[argv.index("--output-dir") + 1] == b["output_dir"]
        Path(argv[3]).write_text("Maximum resident set size (kbytes): 12345\n")
        files = {}
        for name in b["variants"]:
            path = Path(b["output_dir"]) / name / "H2O1/chunk_00000/ckd.npz"
            path.parent.mkdir(parents=True)
            path.write_bytes(name.encode())
            files[name] = worker.pin(path)
        source, _ = worker.sampling.refinement._source_chunk(b["refinement_plan"], 0)
        worker.sampling._write_json(worker.sampling._chunk_path(b, 0), dict(
            fingerprint=b["fingerprint"], species="H2O1", chunk_index=0,
            files=files, inputs=dict(source_chunk=worker.pin(source))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", child)
    monkeypatch.setattr(worker, "available_gib", lambda: 42.)
    monkeypatch.setattr(worker, "active_main_backlog", lambda plan: [])
    return calls


def test_transfer_preserves_original_bytes_publishes_and_resumes(worker, tmp_path, monkeypatch):
    a, b = fixture_plans(worker, tmp_path)
    original = worker.sampling.refinement._chunk_path(a["refinement_plan"], 0)
    old = original.read_bytes()
    calls = install_child(worker, b, monkeypatch)
    logs = tmp_path / "logs"
    logs.mkdir()
    row = worker.transfer_chunk(a, b, 0, logs, {}, 32., {"code_sources": {}})
    assert row["child"]["max_rss_kib"] == 12345
    assert len(calls) == 1 and original.read_bytes() == old
    copied = worker.sampling.refinement._chunk_path(b["refinement_plan"], 0)
    assert original.samefile(copied)
    assert worker.sampling._chunk_path(a, 0).samefile(worker.sampling._chunk_path(b, 0))
    record = worker.sampling.completed_chunk(a, 0)
    assert all(Path(item["path"]).is_relative_to(b["output_dir"]) for item in record["files"].values())
    row = worker.transfer_chunk(a, b, 0, logs, {}, 32., {"code_sources": {}})
    assert row["status"] == "verified_existing_main" and len(calls) == 1
    # Recover a completed B child that has not yet been published to A.
    worker.sampling._chunk_path(a, 0).unlink()
    row = worker.transfer_chunk(a, b, 0, logs, {}, 32., {"code_sources": {}})
    assert row["child"] is None and len(calls) == 1
    assert original.read_bytes() == old


def test_no_replace_and_wrong_index_are_rejected(worker, tmp_path):
    a, _ = fixture_plans(worker, tmp_path)
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"complete")
    destination.write_bytes(b"other record")
    with pytest.raises(ValueError, match="Refusing to replace"):
        worker.no_replace_link(source, destination)
    assert destination.read_bytes() == b"other record"
    original = worker.sampling.refinement._chunk_path(a["refinement_plan"], 0)
    record = json.loads(original.read_text())
    record["source_chunk"] = worker.pin(source)
    worker.sampling.base._write_json(original, record)
    with pytest.raises(ValueError, match="another chunk"):
        worker.checked_refinement(a, 0)


@pytest.mark.parametrize("guard", ["memory", "overlap"])
def test_guard_prevents_asset_links_and_child_start(worker, tmp_path, monkeypatch, guard):
    a, b = fixture_plans(worker, tmp_path)
    calls = install_child(worker, b, monkeypatch)
    if guard == "memory":
        monkeypatch.setattr(worker, "available_gib", lambda: 31.)
    else:
        monkeypatch.setattr(worker, "active_main_backlog", lambda plan: [{"pid": 123, "chunk_index": 0}])
    with pytest.raises(ValueError, match="available|backlog"):
        worker.transfer_chunk(a, b, 0, tmp_path, {}, 32., {"code_sources": {}})
    assert not Path(b["output_dir"]).exists()
    assert not Path(b["refinement_plan"]["output_dir"]).exists()
    assert not calls


def test_child_failure_preserves_original_and_does_not_publish(worker, tmp_path, monkeypatch):
    a, b = fixture_plans(worker, tmp_path)
    install_child(worker, b, monkeypatch)
    original = worker.sampling.refinement._chunk_path(a["refinement_plan"], 0)
    before = original.read_bytes()

    def failed(argv, **kwargs):
        Path(argv[3]).write_text("Maximum resident set size (kbytes): 42\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(worker.subprocess, "run", failed)
    with pytest.raises(ValueError, match="failed"):
        worker.transfer_chunk(a, b, 0, tmp_path, {}, 32., {"code_sources": {}})
    assert not worker.sampling._chunk_path(a, 0).exists()
    assert original.read_bytes() == before


def test_child_environment_preserves_pythonpath(worker, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/metadata/path:/another/path")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    env = worker.child_environment()
    assert env["PYTHONPATH"] == str(worker.ROOT / "src") + ":/metadata/path:/another/path"
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["JAX_ENABLE_X64"] == "True"
