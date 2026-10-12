"""Check independent endpoint preparation and exact preservation of interior/LBL."""

import argparse
import importlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from tests.integration.offline.examples.test_prepare_rce_wasp18b_ckd_refinement import Teacher, source_fixture


@pytest.fixture
def endpoints(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("prepare_rce_wasp18b_ckd_endpoints")
    jax.config.update("jax_enable_x64", previous)


def make_plan(endpoints, tmp_path):
    source = source_fixture(endpoints.refinement, tmp_path)
    return endpoints.make_plan(argparse.Namespace(
        base_preparation=Path(source["source_preparation_path"]), source_preparation=None,
        species=source["species"], output_dir=tmp_path / "endpoints"), extension_bands=3)


def test_grids_and_dry_run_guard(endpoints, tmp_path, capsys):
    plan = make_plan(endpoints, tmp_path)
    spec = plan["source_plan"]["settings"]
    assert not Path(plan["output_dir"]).exists()
    assert len({item["grid_fingerprint"] for item in plan["variants"].values()}) == 3
    for end in ("lower", "upper"):
        previous = None
        for index in range(plan["chunks_per_end"]):
            padded, nu, crop, edges = endpoints.endpoint_grid(plan, end, index)
            assert len(padded) % 2 == 0
            assert len(nu) == len(edges) * spec["samples_per_band"] + 1
            np.testing.assert_allclose(nu, padded[crop], rtol=2e-15)
            assert padded[0] < nu[0] < nu[-1] < padded[-1]
            np.testing.assert_array_equal(edges[1:, 0], edges[:-1, 1])
            if previous is not None:
                assert previous == edges[0, 0] == nu[0]
            previous = edges[-1, 1]
        if end == "lower":
            assert previous == spec["nu_min"]
        else:
            assert endpoints.endpoint_grid(plan, end, 0)[3][0, 0] == spec["nu_max"]
    arguments = ["--base-preparation", plan["source_preparation_path"], "--species", plan["species"],
                 "--output-dir", plan["output_dir"], "--dry-run"]
    assert endpoints.main(arguments) == 0
    printed = capsys.readouterr().out
    assert '"chunks_per_end"' in printed and not Path(plan["output_dir"]).exists()
    args = argparse.Namespace(base_preparation=Path(plan["source_preparation_path"]),
                              source_preparation=None, species=plan["species"], output_dir=Path(plan["output_dir"]))
    with pytest.raises(ValueError, match="positive integer"):
        endpoints.make_plan(args, extension_bands=0)
    with pytest.raises(ValueError, match="100 cm-1"):
        endpoints.make_plan(args, extension_bands=1000)
    with pytest.raises(ValueError, match="Invalid endpoint"):
        endpoints.endpoint_grid(plan, "lower", -1)


def test_three_variants_reuse_endpoints_and_preserve_original(endpoints, tmp_path, monkeypatch):
    from exojax.opacity import OpaCKD

    plan = make_plan(endpoints, tmp_path)
    base, calls = endpoints.base, []
    source_path = Path(plan["source_preparation_path"])
    original_hash = base._sha256(source_path)
    source = json.loads(source_path.read_text())

    def fake_loader(original, index):
        _, nu, _, edges = base.chunk_grid(original, index)
        directory = Path(original["output_dir"]) / original["species"] / f"chunk_{index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        provenance = dict(sources={}, grid_fingerprint=original["grid_fingerprint"], profile_kernel=None)
        teacher = Teacher(nu, provenance)
        calls.append(teacher)
        spec = original["settings"]
        base.build_tables(teacher, edges, np.asarray(spec["temperature_grid_K"]),
                          np.asarray(spec["pressure_grid_bar"]), spec["ng"], directory, provenance)
        base._write_json(directory / "complete.json", {"wrong_completion": True})

    monkeypatch.setattr(base, "prepare_chunk", fake_loader)
    for end in ("lower", "upper"):
        for index in range(plan["chunks_per_end"]):
            endpoints.prepare_chunk(plan, end, index)
    assert len(calls) == 4 and all(teacher.calls == 1 for teacher in calls)
    assert all(endpoints.completed_chunks(plan, "lower"))
    paths = {name: endpoints.merge_variant(plan, name) for name in endpoints.VARIANTS}
    assert len(calls) == 4 and base._sha256(source_path) == original_hash
    assert not list(Path(plan["output_dir"]).rglob("diffgrid*"))
    tables, records = {}, {}
    with np.load(source["files"]["ckd.npz"]["path"]) as archive:
        original = {key: archive[key] for key in ("band_edges", "nu_bands", "log_kggrid")}
    for name, path in paths.items():
        record = records[name] = json.loads(path.read_text())
        tables[name] = OpaCKD.load_only().load_tables(record["files"]["ckd.npz"]["path"])
        offset = 3 if name != "upper" else 0
        for key, value in original.items():
            actual = np.asarray(getattr(tables[name].ckd_info, key))
            selected = actual[..., offset:offset + len(original["band_edges"])] if key == "log_kggrid" else actual[offset:offset + len(original["band_edges"])]
            np.testing.assert_array_equal(selected, value)
        assert record["original_resolved_files"]["diffgrid.npz"] == source["files"]["diffgrid.npz"]
    np.testing.assert_array_equal(tables["lower"].ckd_info.log_kggrid[..., :3], tables["both"].ckd_info.log_kggrid[..., :3])
    np.testing.assert_array_equal(tables["upper"].ckd_info.log_kggrid[..., -3:], tables["both"].ckd_info.log_kggrid[..., -3:])

    baseline_path = base.assemble_manifest(plan["source_plan"], [plan["species"]])
    baseline = json.loads(baseline_path.read_text())
    star = tmp_path / "star.npz"
    np.savez(star, nu_cm1=[4000., 4700.], surface_flux_per_cm1=[1., 1.])
    baseline["files"]["star"] = base._file_entry(star, "Synthetic wide stellar spectrum")
    baseline["stellar"] = dict(format="npz", version="synthetic", asset="star", flux_unit="erg s-1 cm-2 (cm-1)-1")
    baseline_path.write_text(json.dumps(baseline))
    for name in paths:
        output = tmp_path / f"manifest_{name}.json"
        endpoints.assemble_manifest(baseline_path, plan["output_dir"], name, output)
        assembled = json.loads(output.read_text())
        assert assembled["opacity"]["lbl"] == baseline["opacity"]["lbl"]
        for key in ("C1O1_lbl", "C1O1_lbl_meta", "C1O1_preparation"):
            assert assembled["files"][key] == baseline["files"][key]
        with pytest.raises(ValueError, match="new manifest"):
            endpoints.assemble_manifest(baseline_path, plan["output_dir"], name, output)
    np.savez(star, nu_cm1=[4300., 4400.], surface_flux_per_cm1=[1., 1.])
    baseline["files"]["star"] = base._file_entry(star, "Insufficient stellar range")
    baseline_path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="does not cover"):
        endpoints.assemble_manifest(baseline_path, plan["output_dir"], "both", tmp_path / "short_star.json")
    changed = Path(records["lower"]["endpoint_records"][0]["path"])
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError, match="Changed or missing"):
        endpoints.assemble_manifest(baseline_path, plan["output_dir"], "lower", tmp_path / "changed_chunk.json")
    Path(records["upper"]["files"]["ckd.npz"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        endpoints.assemble_manifest(baseline_path, plan["output_dir"], "upper", tmp_path / "changed_table.json")
