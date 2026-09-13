"""Saved real-line example and failure-report contracts, with no downloads."""

import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return (importlib.import_module("_ckd_mixture_case"),
            importlib.import_module("ckd_mixture_validation"))


def test_real_co_h2o_saved_forward_and_integrity(tmp_path, modules, monkeypatch):
    from exojax.test import data

    case, _ = modules
    # Wheel installs place bundled data outside the example's checkout.
    installed_data = tmp_path / "installed-data"
    for entry in ("CO/12C-16O/SAMPLE", "H2O/1H2-16O/SAMPLE"):
        shutil.copytree(data.get_testdata_filename(entry), installed_data / entry)
    monkeypatch.setattr(data, "get_testdata_filename", lambda entry: installed_data / entry)
    directory = tmp_path / "real-case"
    context = case.prepare_case(directory, samples_per_band=16, ng=4,
                                temperature_nodes=3, validation_points=1)
    assert context["metadata"]["species_order"] == ["CO", "H2O"]
    assert [entry["lines"] for entry in context["metadata"]["molecules"]] == [259, 197]
    assert [entry["source"] for entry in context["metadata"]["molecules"]] == [
        str(installed_data / entry) for entry in ("CO/12C-16O/SAMPLE", "H2O/1H2-16O/SAMPLE")
    ]
    restored = case.load_context(directory)
    for method in case.METHODS:
        forward = jax.jit(case.make_forward(context, method))
        restored_forward = jax.jit(case.make_forward(restored, method))
        value = forward(context["truth"])
        np.testing.assert_array_equal(value, restored_forward(context["truth"]))
        jacobian = jax.jacfwd(forward)(context["truth"])
        assert value.shape == (8,)
        assert np.all(np.isfinite(jacobian))
        assert np.all(np.max(np.abs(jacobian), axis=0) > 1e-6)
    # These tiny grids exercise real input and shared RT; they cannot certify accuracy.
    assert context["metadata"]["budgets"]["max_error_in_noise"] == 0.01
    (directory / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        case.load_context(directory)


def test_validation_separates_approximation_from_local_gradients(tmp_path, modules, monkeypatch):
    case, validation = modules
    context = {
        "bounds": jnp.array([[-1.0, 1.0]] * 3),
        "sigma": jnp.ones(3), "observed": jnp.ones(3),
        "arrays": {"validation_points": jnp.array([[0.1, -0.2, 0.3]])},
        "metadata": {"config": {"ng_values": [2]},
                     "budgets": {"max_error_in_noise": 0.01, "max_q": 0.1,
                                 "gradient_tolerance": 1e-3, "reference_fraction": 0.1,
                                 "steps": [1e-3, 1e-4, 1e-5]}},
    }

    def synthetic_forward(context, method, **kwargs):
        offset = 1.0 if method == "same_g" else 0.0
        return lambda theta: jnp.exp(theta) + offset

    monkeypatch.setattr(validation, "make_forward", synthetic_forward)
    report, arrays = validation.evaluate_context(context)
    assert report["reference"]["passed"]
    assert report["methods"]["rorr"]["passed"]
    assert report["methods"]["same_g"]["gradient_passed"]
    assert not report["methods"]["same_g"]["spectrum_passed"]
    assert not report["methods"]["same_g"]["passed"]
    assert arrays["jacobian_lbl"].shape == (1, 3, 3)

    context.update(case_sha256="case", model_code_sha256="model")
    monkeypatch.setattr(validation, "load_context", lambda _: context)
    monkeypatch.setattr(validation, "collect_provenance", lambda *args: {})
    args = SimpleNamespace(output_dir=tmp_path, validation_id="failure", methods=["same_g"])
    saved = validation.validate_case(args)
    assert saved["status"] == "completed" and not saved["passed"]
    path = tmp_path / "validations" / "failure"
    assert json.loads((path / "validation.json").read_text())["passed"] is False
    assert case.sha256(path / saved["residuals"]["filename"]) == saved["residuals"]["sha256"]
    with pytest.raises(FileExistsError):
        validation.validate_case(args)


def test_source_and_frozen_metadata_changes_are_rejected(tmp_path, modules, monkeypatch):
    case, _ = modules
    (tmp_path / "case.json").write_text('{"budgets": {"max_error_in_noise": 0.01}}')
    case.write_npz(tmp_path / "arrays.npz", values=np.ones(2))
    manifest = {"schema_version": 1, "status": "completed", "model_code_sha256": "original",
                "artifacts": {name: case.sha256(tmp_path / name) for name in ("case.json", "arrays.npz")}}
    case.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(case, "model_code_sha256", lambda: "changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        case.verify_case(tmp_path)
    monkeypatch.setattr(case, "model_code_sha256", lambda: "original")
    (tmp_path / "case.json").write_text('{"budgets": {"max_error_in_noise": 1.0}}')
    with pytest.raises(ValueError, match="hash mismatch: case.json"):
        case.verify_case(tmp_path)


def test_cli_preserves_frozen_budgets_and_rejects_invalid_method(modules):
    _, validation = modules
    args = validation.parser().parse_args(["validate", "--output-dir", "case",
                                           "--validation-id", "check"])
    assert args.methods == ["lbl", "rorr"]
    with pytest.raises(SystemExit):
        validation.parser().parse_args(["validate", "--output-dir", "case",
                                        "--validation-id", "check", "--methods", "unknown"])
