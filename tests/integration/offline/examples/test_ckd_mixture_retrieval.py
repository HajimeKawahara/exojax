"""Offline evidence and matched-prior contracts for CKD mixture retrievals."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def retrieval(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("ckd_mixture_retrieval")


@pytest.fixture
def case(retrieval, monkeypatch, tmp_path):
    """Real hashes and archives, with tiny arrays and no molecular database load."""
    storage = retrieval.common._storage()
    directory = tmp_path / "case"
    directory.mkdir()
    monkeypatch.setattr(retrieval.mixture, "model_code_sha256", lambda: "fixture-model")
    bounds = np.array([[-9.0, -4.0], [-8.0, -3.0], [-0.1, 0.1]])
    storage.write_json(directory / "case.json", {
        "schema_version": 1, "parameter_order": list(retrieval.mixture.PARAMETER_ORDER),
        "species_order": ["CO", "H2O"], "config": {"ng": 4, "ng_values": [2, 4, 8]},
    })
    storage.write_npz(directory / "arrays.npz", bounds=bounds, truth=bounds.mean(axis=1),
                      observed=np.zeros(4), sigma=np.ones(4), nu_bands=np.arange(4.) + 4300)
    storage.write_json(directory / "manifest.json", {
        "schema_version": 1, "status": "completed", "model_code_sha256": "fixture-model",
        "artifacts": {name: storage.sha256(directory / name) for name in ("case.json", "arrays.npz")},
    })
    validation_dir = directory / "validations" / "accuracy"
    validation_dir.mkdir(parents=True)
    storage.write_npz(validation_dir / "spectra.npz", parameters=bounds.mean(axis=1)[None],
                      reference_refined=np.zeros((1, 4)), sigma=np.ones(4),
                      spectrum_lbl=np.zeros((1, 4)), spectrum_rorr=np.zeros((1, 4)))
    storage.write_json(validation_dir / "validation.json", {
        "schema_version": 1, "status": "completed", "validation_id": "accuracy",
        "case_sha256": storage.sha256(directory / "manifest.json"),
        "model_code_sha256": "fixture-model", "environment": retrieval.mixture.runtime_environment(),
        "reference": {"passed": True}, "passed": True,
        "methods": {method: {"passed": True, "gradient_passed": True} for method in retrieval.mixture.METHODS},
        "residuals": {"filename": "spectra.npz", "sha256": storage.sha256(validation_dir / "spectra.npz")},
    })
    return directory, bounds


def _arguments(retrieval, directory, method="lbl", run_id="pair_a", seed=17, **settings):
    arguments = ["run", "--output-dir", str(directory), "--method", method,
                 "--run-id", run_id, "--validation-id", "accuracy", "--seed", str(seed)]
    for name, value in settings.items():
        arguments.extend(["--" + name.replace("_", "-"), str(value)])
    return retrieval._parser().parse_args(arguments)


def _update(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))
    return value


def _saved_run(retrieval, case, method="lbl", run_id="pair_a", seed=17, divergences=0, diagnostic=False):
    """Create four IID chains so quality checks use real ArviZ diagnostics."""
    directory, bounds = case
    args = _arguments(retrieval, directory, method, run_id, seed)
    storage = retrieval.common._storage()
    destination = retrieval._directory(directory, run_id, method)
    destination.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    samples = {name: rng.uniform(low, high, (4, 1000))
               for name, (low, high) in zip(retrieval.mixture.PARAMETER_ORDER, bounds)}
    extra = {"diverging": np.zeros((4, 1000), dtype=bool),
             "num_steps": np.full((4, 1000), 7), "accept_prob": np.full((4, 1000), 0.9)}
    extra["diverging"].flat[:divergences] = True
    manifest = storage.save_samples(destination / "samples.npz", samples, extra,
                                    retrieval.mixture.PARAMETER_ORDER)
    prediction = destination / "posterior_predictive.npz"
    storage.write_npz(prediction, prediction=np.zeros((4, 2, 4)), draw_indices=np.array([0, 999]),
                      observation=np.zeros(4), sigma=np.ones(4), nu_bands=np.arange(4.) + 4300)
    validation = retrieval._validation(directory, "accuracy", method,
                                       {"case_sha256": storage.sha256(directory / "manifest.json")},
                                       retrieval.mixture.runtime_environment(), diagnostic=diagnostic)
    _, _, _, initialization = retrieval._initialization(bounds, args)
    result = {
        "schema_version": 2, "status": "completed", "stage": "completed",
        "run_id": run_id, "method": method, "samples": manifest,
        "case_sha256": storage.sha256(directory / "manifest.json"),
        "case_metadata_sha256": storage.sha256(directory / "case.json"),
        "validation": validation, "initialization": initialization,
        "environment": retrieval.mixture.runtime_environment(),
        "sampler_environment": retrieval.common._environment(),
        "provenance": {"code_sha256": "fixture-code", "dependencies": {}, "environment": {}},
        "quality_rules": dict(retrieval.common._helper("benchmark_inference").DEFAULT_RULES),
        "settings": {"seed": seed, "num_chains": 4, "num_samples": 1000, "num_warmup": 500,
                     "predictive_draws": 2, "diagnostic": diagnostic, "chain_method": "sequential", "dense_mass": True,
                     "target_accept_probability": 0.9, "max_tree_depth": 10},
        "timings": {"sampling_compile_and_run_seconds": 20.0 if method == "lbl" else 5.0},
        "posterior_predictive": {"sha256": storage.sha256(prediction), "shape": [4, 2, 4], "finite": True},
        # This forged success is deliberately ignored when reading saved evidence.
        "posterior_inference": {"quality_passed": True},
    }
    storage.write_json(destination / "result.json", result)
    return destination


def test_initialization_is_method_independent_and_repeats_are_independent(retrieval, case):
    directory, bounds = case
    first = retrieval._initialization(bounds, _arguments(retrieval, directory, "lbl"))
    second = retrieval._initialization(bounds, _arguments(retrieval, directory, "rorr"))
    assert first[3] == second[3]
    for name in retrieval.mixture.PARAMETER_ORDER:
        np.testing.assert_array_equal(first[0][name], second[0][name])
    fractions = (np.asarray(first[3]["positions"]) - bounds[:, 0]) / np.diff(bounds, axis=1)[:, 0]
    assert np.all((fractions >= 0.05) & (fractions <= 0.95))
    assert first[3]["parameter_order"] == list(retrieval.mixture.PARAMETER_ORDER)
    for i, name in enumerate(retrieval.mixture.PARAMETER_ORDER):
        np.testing.assert_allclose(first[0][name], np.log(fractions[:, i] / (1 - fractions[:, i])))
    repeat = retrieval._initialization(bounds, _arguments(retrieval, directory, seed=18))
    assert repeat[3]["positions"] != first[3]["positions"]
    keys = [tuple(key) for record in (first[3], repeat[3])
            for group in ("warmup_keys", "sampling_keys") for key in record[group]]
    assert len(set(keys)) == 16


def test_models_share_the_same_normalized_prior(retrieval, case):
    pytest.importorskip("numpyro")
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    _, bounds = case
    theta = bounds.mean(axis=1)
    parameters = dict(zip(retrieval.mixture.PARAMETER_ORDER, theta))
    observation = jnp.array([0.1, -0.2])
    sigma = jnp.array([0.3, 0.4])
    prior = -np.log(np.diff(bounds, axis=1)[:, 0]).sum()
    for offset in (0.0, 0.5):
        model = retrieval.make_model(lambda p: p[:2] + offset, bounds, sigma)
        joint, trace = log_density(model, (observation,), {}, parameters)
        likelihood = trace["spectrum"]["fn"].log_prob(observation).sum()
        np.testing.assert_allclose(joint - likelihood, prior, atol=1.e-12)
        assert [name for name in trace if name != "spectrum"] == list(retrieval.mixture.PARAMETER_ORDER)


@pytest.mark.parametrize("target", ["samples.npz", "posterior_predictive.npz", "validation", "case"])
def test_load_rejects_changed_evidence(retrieval, case, target):
    directory, _ = case
    destination = _saved_run(retrieval, case)
    path = {"validation": directory / "validations/accuracy/validation.json",
            "case": directory / "case.json"}.get(target, destination / target)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest|changed"):
        retrieval._load_run(directory, "pair_a", "lbl")


@pytest.mark.parametrize("status, schema", [("partial", 2), ("failed", 2), ("completed", 1)])
def test_load_rejects_partial_or_legacy_run_evidence(retrieval, case, status, schema):
    directory, _ = case
    destination = _saved_run(retrieval, case)
    _update(destination / "result.json", status=status, schema_version=schema)
    with pytest.raises(ValueError, match="completed|current schema"):
        retrieval._load_run(directory, "pair_a", "lbl")


def test_load_rejects_prediction_shape_inconsistent_with_primary_chains(retrieval, case):
    directory, _ = case
    destination = _saved_run(retrieval, case)
    storage = retrieval.common._storage()
    prediction = destination / "posterior_predictive.npz"
    with np.load(prediction, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["prediction"] = arrays["prediction"][:1]
    storage.write_npz(prediction, **arrays)
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text())
    result["posterior_predictive"].update(sha256=storage.sha256(prediction), shape=[1, 2, 4])
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="posterior|prediction|Prediction"):
        retrieval._load_run(directory, "pair_a", "lbl")


@pytest.mark.parametrize("change", ["partial", "failed_method", "code", "environment", "residual"])
def test_validation_gate_rejects_invalid_evidence(retrieval, case, change):
    directory, _ = case
    path = directory / "validations/accuracy/validation.json"
    value = json.loads(path.read_text())
    if change == "partial":
        value["status"] = "partial"
    elif change == "failed_method":
        value["methods"]["rorr"]["passed"] = False
    elif change == "code":
        value["model_code_sha256"] = "another-model"
    elif change == "environment":
        value["environment"]["backend"] = "another-backend"
    else:
        (path.parent / "spectra.npz").write_bytes(b"changed archive")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        retrieval._validation(directory, "accuracy", "rorr",
                              {"case_sha256": retrieval.common._storage().sha256(directory / "manifest.json")},
                              retrieval.mixture.runtime_environment())


def _summary(retrieval, directory, repeats=("pair_b",)):
    args = retrieval._parser().parse_args([
        "summarize", "--output-dir", str(directory), "--run-id", "pair_a",
        *[value for run_id in repeats for value in ("--repeat-run-id", run_id)],
    ])
    retrieval.summarize(args)
    return json.loads((directory / "runs/pair_a/comparison.json").read_text())


@pytest.mark.parametrize("failure", ["spectrum", "reference", "gradient"])
def test_diagnostic_mode_only_allows_spectral_approximation_failure(retrieval, case, failure):
    directory, _ = case
    path = directory / "validations/accuracy/validation.json"
    evidence = json.loads(path.read_text())
    evidence["methods"]["rorr"]["passed"] = False
    if failure == "reference":
        evidence["reference"]["passed"] = False
    if failure == "gradient":
        evidence["methods"]["rorr"]["gradient_passed"] = False
    path.write_text(json.dumps(evidence))
    arguments = (directory, "accuracy", "rorr",
                 {"case_sha256": retrieval.common._storage().sha256(directory / "manifest.json")},
                 retrieval.mixture.runtime_environment())
    with pytest.raises(ValueError):
        retrieval._validation(*arguments)
    if failure == "spectrum":
        record = retrieval._validation(*arguments, diagnostic=True)
        assert record["diagnostic"] and not record["passed"]
    else:
        with pytest.raises(ValueError, match="Reference refinement and local gradient"):
            retrieval._validation(*arguments, diagnostic=True)


def test_converged_diagnostic_retrieval_never_qualifies_as_scientific_speedup(retrieval, case):
    pytest.importorskip("arviz")
    directory, _ = case
    path = directory / "validations/accuracy/validation.json"
    evidence = json.loads(path.read_text())
    evidence["methods"]["rorr"]["passed"] = False
    path.write_text(json.dumps(evidence))
    for run_id, seed in (("pair_a", 17), ("pair_b", 18)):
        for method in ("lbl", "rorr"):
            _saved_run(retrieval, case, method, run_id, seed, diagnostic=True)
    summary = _summary(retrieval, directory)
    assert not summary["quality"]["eligible"]
    assert all("diagnostic retrieval" in reason for reason in summary["quality"]["reasons"])
    assert summary["cold_sampling_speedups"]["rorr"]["scientific_median"] is None


@pytest.mark.parametrize("repetitions, divergences", [(2, 0), (1, 0), (2, 1)])
def test_summary_requires_independent_pairs_and_recomputes_quality(retrieval, case, repetitions, divergences):
    pytest.importorskip("arviz")
    directory, _ = case
    for run_id, seed in [("pair_a", 17), ("pair_b", 18)][:repetitions]:
        for method in ("lbl", "rorr"):
            _saved_run(retrieval, case, method, run_id, seed, divergences)
    result = _summary(retrieval, directory, ("pair_b",) if repetitions == 2 else ())
    assert result["status"] == "completed"
    eligible = repetitions == 2 and divergences == 0
    assert result["quality"]["eligible"] is eligible
    assert result["cold_sampling_speedups"]["rorr"]["scientific_median"] == (4.0 if eligible else None)
    if divergences:
        assert not result["repetitions"][0]["lbl"]["posterior_inference"]["quality_passed"]


@pytest.mark.parametrize("change", ["same_seed", "same_initialization_seed", "sampling_keys", "controls",
                                    "environment", "provenance", "validation"])
def test_summary_rejects_unmatched_or_dependent_runs(retrieval, case, change):
    directory, _ = case
    destinations = {}
    for run_id, seed in [("pair_a", 17), ("pair_b", 18)]:
        for method in ("lbl", "rorr"):
            destinations[run_id, method] = _saved_run(retrieval, case, method, run_id, seed)
    for method in ("lbl", "rorr"):
        path = destinations["pair_b", method] / "result.json"
        value = json.loads(path.read_text())
        if change == "same_seed":
            value["settings"]["seed"] = 17
        elif change == "same_initialization_seed":
            value["initialization"]["seed"] = 17
        elif change == "sampling_keys":
            first = json.loads((destinations["pair_a", method] / "result.json").read_text())
            value["initialization"]["sampling_keys"] = first["initialization"]["sampling_keys"]
        elif change == "controls":
            value["settings"]["num_warmup"] = 600
        elif change == "environment":
            value["sampler_environment"]["python"] = "another-version"
        elif change == "provenance":
            value["provenance"]["code_sha256"] = "another-code"
        else:
            value["validation"]["sha256"] = "another-validation"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        _summary(retrieval, directory)
    assert json.loads((directory / "runs/pair_a/comparison.json").read_text())["status"] == "failed"


def test_cpu_sampler_persists_matched_inputs_raw_chains_and_failed_smoke_quality(retrieval, case, monkeypatch):
    pytest.importorskip("numpyro")
    import jax.numpy as jnp

    directory, _ = case
    monkeypatch.setattr(retrieval, "_provenance", lambda _: {
        "code_sha256": "fixture-code", "dependencies": {}, "environment": {},
    })
    monkeypatch.setattr(retrieval.mixture, "make_forward", lambda *_: lambda p: jnp.concatenate((p, p[:1])))
    records, saved_samples = [], []
    for method in ("lbl", "rorr"):
        args = _arguments(retrieval, directory, method, num_chains=2,
                          num_warmup=4, num_samples=8, predictive_draws=2)
        retrieval.run(args)
        result = retrieval._load_run(directory, "pair_a", method)
        assert result["status"] == "completed"
        assert result["samples"]["chain_shape"] == [2, 8]
        assert result["posterior_predictive"]["shape"] == [2, 2, 4]
        assert not result["posterior_inference"]["quality_passed"]
        samples, _ = retrieval.common._storage().load_samples(
            directory / "runs/pair_a" / method / "samples.npz", result["samples"]
        )
        records.append(result)
        saved_samples.append(samples)
    assert records[0]["settings"] == records[1]["settings"]
    assert records[0]["initialization"] == records[1]["initialization"]
    assert records[0]["validation"] == records[1]["validation"]
    for name in retrieval.mixture.PARAMETER_ORDER:
        np.testing.assert_array_equal(saved_samples[0][name], saved_samples[1][name])
