"""Independent and pathological chain checks for optional posterior diagnostics."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def inference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmark"))
    return importlib.import_module("benchmark_inference")


@pytest.fixture
def arviz():
    return pytest.importorskip("arviz")


def _extra(shape):
    return {"diverging": np.zeros(shape, dtype=bool)}


def _without_arviz(monkeypatch, inference):
    original = inference.importlib.import_module

    def load(name, *args, **kwargs):
        if name == "arviz":
            raise ModuleNotFoundError("ArviZ is intentionally absent.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(inference.importlib, "import_module", load)


def test_independent_normal_chains_recover_moments_and_monte_carlo_error(
    inference, arviz
):
    values = np.random.default_rng(5729).normal(size=(4, 1500))
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert report["quality_passed"]
    assert report["failure_reasons"] == []
    record = report["per_parameter"]["x"]
    assert record["mean"] == pytest.approx(0, abs=0.06)
    assert record["quantiles"]["0.05"] == pytest.approx(-1.64485, abs=0.09)
    assert record["quantiles"]["0.5"] == pytest.approx(0, abs=0.06)
    assert record["quantiles"]["0.95"] == pytest.approx(1.64485, abs=0.09)
    independent_mcse = 1 / np.sqrt(values.size)
    assert 0.7 * independent_mcse < record["mcse_mean"] < 1.3 * independent_mcse
    assert all(value > 0 for value in record["mcse_quantiles"].values())
    assert record["rhat_rank"] < 1.01
    assert record["ess_bulk"] >= 400 and record["ess_tail"] >= 400
    assert report["implementation"]["version"] == arviz.__version__
    assert "folded" in report["definitions"]["rhat_rank"]
    assert "0.05, 0.95" in report["definitions"]["ess_tail"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pathology", ["separated", "different_scales", "correlated"])
def test_chains_with_poor_mixing_fail_quality(inference, arviz, pathology):
    values = np.random.default_rng(31).normal(size=(4, 1000))
    if pathology == "separated":
        values += np.arange(4)[:, None] * 3
    elif pathology == "different_scales":
        values *= np.array([1, 1, 10, 10])[:, None]
    else:
        for index in range(1, values.shape[1]):
            values[:, index] += 0.98 * values[:, index - 1]
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert not report["quality_passed"]
    record = report["per_parameter"]["x"]
    if pathology == "correlated":
        assert record["ess_bulk"] < 400
    else:
        assert record["rhat_rank"] > 1.1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("all_chains", [True, False])
def test_constant_or_stuck_chain_cannot_pass_quality(inference, arviz, all_chains):
    values = np.random.default_rng(7).normal(size=(4, 1000))
    values[:] = 2.0 if all_chains else values
    values[0] = 2.0
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert not report["quality_passed"]
    assert report["per_parameter"]["x"]["constant_chains"] == (
        [0, 1, 2, 3] if all_chains else [0]
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e308])
def test_nonfinite_samples_and_overflow_are_json_safe(inference, arviz, value):
    values = np.ones((4, 20))
    values[:, 3] = value
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert not report["quality_passed"]
    assert report["per_parameter"]["x"]["mean"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("shape", [(1, 3), (4, 3), (1, 20)])
def test_smoke_chains_are_saved_without_claiming_scientific_convergence(
    inference, arviz, shape
):
    values = np.random.default_rng(43).normal(size=shape)
    report = inference.posterior_diagnostics({"x": values}, _extra(shape))
    assert report["status"] == "completed"
    assert not report["quality_passed"]
    assert report["per_parameter"]["x"]["mean"] is not None
    assert report["summary"]["max_rhat_rank"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("flags", [None, [False], [[np.nan]], [[2]]])
def test_missing_or_malformed_divergences_explicitly_fail(
    inference, monkeypatch, flags
):
    _without_arviz(monkeypatch, inference)
    values = np.arange(80.0).reshape(4, 20)
    if flags is None:
        extra = {}
    else:
        extra = {
            "diverging": np.broadcast_to(flags, (4, 20))
            if len(flags) == 1 and isinstance(flags[0], list)
            else flags
        }
    report = inference.posterior_diagnostics({"x": values}, extra)
    assert not report["quality_passed"]
    assert report["summary"]["divergences"] is None
    assert any("Divergence" in message for message in report["failure_reasons"])
    json.dumps(report, allow_nan=False)


def test_divergence_prevents_otherwise_good_chains_from_passing(inference, arviz):
    values = np.random.default_rng(5729).normal(size=(4, 1500))
    extra = _extra(values.shape)
    extra["diverging"][2, 3] = True
    report = inference.posterior_diagnostics({"x": values}, extra)
    assert not report["quality_passed"]
    assert report["summary"]["divergences"] == 1
    assert report["per_parameter"]["x"]["quality_passed"]


def test_missing_optional_library_keeps_descriptive_summary_but_fails_quality(
    inference, monkeypatch
):
    _without_arviz(monkeypatch, inference)
    values = np.arange(80.0).reshape(4, 20)
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert report["status"] == "unavailable" and not report["quality_passed"]
    assert report["implementation"] == {
        "name": "arviz",
        "version": None,
        "available": False,
    }
    assert report["per_parameter"]["x"]["mean"] == 39.5
    assert report["summary"]["min_ess_bulk"] is None
    assert report["summary"]["min_ess_tail"] is None
    assert any("intentionally absent" in reason for reason in report["failure_reasons"])
    json.dumps(report, allow_nan=False)


def test_import_does_not_load_optional_runtime_or_change_cwd_or_jax(inference):
    source = """
import importlib.util, os, sys
import jax
cwd = os.getcwd()
dtype = jax.config.x64_enabled
spec = importlib.util.spec_from_file_location('isolated_inference', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert os.getcwd() == cwd
assert jax.config.x64_enabled == dtype
assert 'arviz' not in sys.modules
assert 'numpyro' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source, inference.__file__], check=True)


def test_quality_rules_use_strict_rhat_and_inclusive_ess(inference, monkeypatch):
    fake_arviz = SimpleNamespace(
        __version__="test",
        rhat=lambda values, **kwargs: 1.01,
        ess=lambda values, **kwargs: 400.0,
        mcse=lambda values, **kwargs: 0.1,
    )
    monkeypatch.setattr(inference.importlib, "import_module", lambda name: fake_arviz)
    values = np.arange(80.0).reshape(4, 20)
    report = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    assert not report["quality_passed"]
    assert report["per_parameter"]["x"]["failure_reasons"] == [
        "Rank R-hat reaches or exceeds its strict upper limit."
    ]
    fake_arviz.rhat = lambda values, **kwargs: 1.0099
    assert inference.posterior_diagnostics({"x": values}, _extra(values.shape))[
        "quality_passed"
    ]


@pytest.mark.parametrize(
    "rules",
    [
        {"unknown": 3},
        {"min_chains": 1},
        {"max_rhat": np.inf},
        {"min_ess_bulk": -1},
        {"max_divergences": 0.5},
        {"min_chains": True},
    ],
)
def test_invalid_quality_rules_are_rejected(inference, rules):
    with pytest.raises(ValueError, match="quality rule"):
        inference.posterior_diagnostics({"x": np.ones((4, 10))}, {}, rules=rules)


def test_vector_parameters_keep_each_component_and_shared_shape(inference, arviz):
    values = np.random.default_rng(9).normal(size=(4, 1000, 2))
    report = inference.posterior_diagnostics({"x": values}, _extra((4, 1000)))
    assert set(report["per_parameter"]) == {"x[0]", "x[1]"}
    assert report["per_parameter"]["x[1]"]["mean"] == pytest.approx(
        values[:, :, 1].mean()
    )
    with pytest.raises(ValueError, match="dimensions"):
        inference.posterior_diagnostics({"x": values, "y": np.ones((4, 5))}, {})


def test_posterior_difference_tracks_shift_and_mcse_without_equivalence_claim(
    inference, arviz
):
    values = np.random.default_rng(13).normal(size=(4, 1000))
    left = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    right = inference.posterior_diagnostics({"x": values + 2}, _extra(values.shape))
    result = inference.posterior_difference(left, right)
    record = result["per_parameter"]["x"]
    for metric in (record["mean"], *record["quantiles"].values()):
        assert metric["difference"] == pytest.approx(2.0)
        assert metric["combined_mcse"] > 0
        assert metric["standardized_difference"] == pytest.approx(
            2 / metric["combined_mcse"]
        )
    assert record["mean"]["combined_mcse"] == pytest.approx(
        np.sqrt(2) * left["per_parameter"]["x"]["mcse_mean"]
    )
    assert "Shared random seeds" in result["mcse_definition"]
    assert "passed" not in result
    json.dumps(result, allow_nan=False)


def test_unavailable_mcse_and_parameter_mismatch_are_not_hidden(inference, monkeypatch):
    _without_arviz(monkeypatch, inference)
    values = np.arange(80.0).reshape(4, 20)
    left = inference.posterior_diagnostics({"x": values}, _extra(values.shape))
    result = inference.posterior_difference(left, left)
    assert result["per_parameter"]["x"]["mean"] == {
        "difference": 0.0,
        "combined_mcse": None,
        "standardized_difference": None,
    }
    with pytest.raises(ValueError, match="identical"):
        inference.posterior_difference(left, {"per_parameter": {}})


def test_posterior_model_prediction_summary_preserves_observation_axis(inference):
    draws = np.arange(8.0).reshape(2, 4)
    predictions = np.stack((draws, 2 * draws + 1), axis=-1)
    report = inference.predictive_summary(predictions)
    assert report["finite"] and report["shape"] == [2, 4, 2]
    assert report["mean"] == [3.5, 8.0]
    assert report["quantiles"]["0.05"] == pytest.approx([0.35, 1.7])
    assert report["quantiles"]["0.5"] == [3.5, 8.0]
    assert report["quantiles"]["0.95"] == pytest.approx([6.65, 14.3])
    assert "measurement noise is not added" in report["definition"]
    predictions[1, 2, 1] = np.nan
    failed = inference.predictive_summary(predictions)
    assert not failed["finite"] and failed["nonfinite_count"] == 1
    assert failed["mean"] is None
    json.dumps(failed, allow_nan=False)
    with pytest.raises(ValueError, match="chain, draw, observation"):
        inference.predictive_summary(draws)
