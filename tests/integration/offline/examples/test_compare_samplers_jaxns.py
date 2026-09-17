"""Weighted-sample contracts and optional real JAXNS CPU smoke tests."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "examples" / "_compare_samplers_jaxns.py"
SPEC = importlib.util.spec_from_file_location("_test_jaxns_adapter", MODULE_PATH)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def fake_results(reason=4):
    return SimpleNamespace(
        samples={"x": np.array([-1.0, 0.0, 1.0])},
        log_dp_mean=np.log([0.1, 0.2, 0.7]),
        log_L_samples=np.array([-2.0, -1.0, -2.0]),
        U_samples=np.array([[0.2], [0.5], [0.8]]),
        log_X_mean=np.array([-1.0, -2.0, -3.0]),
        num_live_points_per_sample=np.array([8, 8, 7]),
        num_likelihood_evaluations_per_sample=np.array([1, 2, 3]),
        total_num_samples=3,
        log_Z_mean=-2.0,
        log_Z_uncert=0.1,
        ESS=1.25,
        termination_reason=reason,
        total_num_likelihood_evaluations=6,
        total_phantom_samples=0,
    )


def test_import_does_not_load_optional_stack_or_change_precision():
    script = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('adapter',{str(MODULE_PATH)!r}); "
        "s.loader.exec_module(importlib.util.module_from_spec(s)); "
        "assert not any(n in sys.modules for n in "
        "('jax','jaxns','tensorflow_probability'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_missing_optional_dependency_is_actionable(monkeypatch):
    def missing(_):
        raise adapter.metadata.PackageNotFoundError("jaxns")

    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(ImportError, match="JAXNS is optional"):
        adapter.build_jaxns_model(lambda _: 0.0, {})


def test_untested_jaxns_api_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "9.0.0")
    with pytest.raises(RuntimeError, match="supports jaxns==2.6.9"):
        adapter.build_jaxns_model(lambda _: 0.0, {})


def test_weighted_draws_round_trip_without_resampling(tmp_path):
    results = fake_results()
    report, arrays = adapter._summarize_results(results)
    assert report["weighted_ess"] == pytest.approx(1 / (0.1**2 + 0.2**2 + 0.7**2))
    assert report["weighted_ess"] != report["num_samples"]
    assert report["jaxns_ess"] == 1.25
    assert report["converged"]
    np.testing.assert_array_equal(arrays["log_weights"], results.log_dp_mean)
    path = tmp_path / "nested.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as restored:
        for name, value in arrays.items():
            np.testing.assert_array_equal(restored[name], value)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("reason", [1, 5, 64, 128, 1024, 8192, 0])
def test_exhaustion_or_unknown_stop_does_not_imply_convergence(reason):
    report, _ = adapter._summarize_results(fake_results(reason))
    assert not report["converged"]
    assert report["status"] == "unconverged"
    assert report["termination_reason"] == reason


@pytest.mark.parametrize("field", ["log_Z_mean", "log_Z_uncert", "ESS", "log_dp_mean"])
def test_nonfinite_results_are_failed_and_json_safe(field):
    results = fake_results()
    setattr(results, field, np.full(3, np.nan) if field == "log_dp_mean" else np.nan)
    report, arrays = adapter._summarize_results(results)
    assert not report["converged"]
    assert report["status"] == "failed"
    assert "log_weights" in arrays
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"num_live_points": 1},
        {"max_samples": 128},
        {"dlogz": 0},
        {"dlogz": float("nan")},
    ],
)
def test_invalid_settings_fail_before_optional_import(settings):
    kwargs = {"seed": 0, **settings}
    with pytest.raises(ValueError):
        adapter.run_nested(lambda _: 0.0, {}, **kwargs)


def saved_mixed_results():
    results = fake_results()
    unit = np.asarray([[0.25, 0.2], [0.5, 0.4], [0.75, 0.6]])
    results.U_samples = unit
    results.samples["sigmain"] = -np.log1p(-unit[:, 1]) / 1e-3
    report, arrays = adapter._summarize_results(results)
    report["parameter_order"] = ["x", "sigmain"]
    # Serialized metadata can have a different dictionary order.
    priors = {
        "sigmain": {"distribution": "exponential", "rate": 1e-3},
        "x": {"distribution": "uniform", "low": -2.0, "high": 2.0},
    }
    return report, arrays, priors


def test_saved_result_recomputes_quality_in_explicit_parameter_order(monkeypatch):
    report, arrays, priors = saved_mixed_results()
    report.update(weighted_ess=1e20, converged=False, status="failed")

    def forbidden():
        raise AssertionError("Reloading must not import the optional sampler.")

    monkeypatch.setattr(adapter, "_runtime", forbidden)
    actual = adapter.validate_saved_results(report, arrays, priors)
    assert actual["converged"]
    assert actual["status"] == "completed"
    assert actual["weighted_ess"] == pytest.approx(1 / (0.1**2 + 0.2**2 + 0.7**2))
    assert report["weighted_ess"] == 1e20


@pytest.mark.parametrize(
    "corruption",
    [
        "weights",
        "nan_weights",
        "unit",
        "physical",
        "order",
        "shape",
        "missing",
        "parameters",
    ],
)
def test_saved_result_rejects_inconsistent_raw_arrays(corruption):
    report, arrays, priors = saved_mixed_results()
    if corruption == "weights":
        arrays["weights"] = np.array([0.2, 0.2, 0.6])
    elif corruption == "nan_weights":
        arrays["weights"][0] = np.nan
    elif corruption == "unit":
        arrays["unit_samples"][0, 0] += 0.1
    elif corruption == "physical":
        arrays["samples__sigmain"][0] += 10
    elif corruption == "order":
        report["parameter_order"].reverse()
    elif corruption == "shape":
        arrays["samples__x"] = arrays["samples__x"][None, :]
    elif corruption == "missing":
        arrays.pop("log_likelihood")
    else:
        arrays["samples__other"] = np.ones(3)
    with pytest.raises(ValueError):
        adapter.validate_saved_results(report, arrays, priors)


@pytest.mark.parametrize(
    "failure", ["log_weights", "physical", "unit", "evidence", "likelihood"]
)
def test_saved_nonfinite_run_remains_failed_with_json_safe_diagnostics(failure):
    report, arrays, priors = saved_mixed_results()
    if failure == "log_weights":
        arrays["log_weights"][0] = np.nan
        arrays["weights"][:] = np.nan
    elif failure == "physical":
        arrays["samples__x"][0] = np.nan
    elif failure == "unit":
        arrays["unit_samples"][0, 0] = np.nan
    elif failure == "likelihood":
        arrays["log_likelihood"][0] = -np.inf
    else:
        report["log_evidence"] = None
    actual = adapter.validate_saved_results(report, arrays, priors)
    assert actual["status"] == "failed"
    assert not actual["converged"]
    for name in ("weighted_ess", "jaxns_ess", "log_evidence", "log_evidence_error"):
        assert actual[name] is None
    json.dumps(actual, allow_nan=False)


def test_saved_zero_weight_negative_infinite_likelihood_is_valid():
    report, arrays, priors = saved_mixed_results()
    arrays["log_weights"] = np.array([-np.inf, np.log(0.25), np.log(0.75)])
    arrays["weights"] = np.array([0.0, 0.25, 0.75])
    arrays["log_likelihood"][0] = -np.inf
    actual = adapter.validate_saved_results(report, arrays, priors)
    assert actual["converged"]
    assert actual["weighted_ess"] == pytest.approx(1.6)


def test_saved_exponential_endpoint_cannot_pass_with_finite_physical_sample():
    report, arrays, priors = saved_mixed_results()
    arrays["unit_samples"][0, 1] = 1.0
    actual = adapter.validate_saved_results(report, arrays, priors)
    assert actual["status"] == "failed"
    assert not actual["converged"]
    assert actual["weighted_ess"] is None


@pytest.fixture
def jaxns_runtime():
    try:
        version = adapter.metadata.version("jaxns")
    except adapter.metadata.PackageNotFoundError:
        pytest.skip("JAXNS is optional; exercised in the separate JAXNS environment.")
    if version != adapter.SUPPORTED_JAXNS_VERSION:
        pytest.skip("The real smoke test requires the explicitly supported JAXNS API.")
    return adapter._runtime()


def test_real_jaxns_mixed_prior_transform_and_physical_density(jaxns_runtime):
    import jax
    import jax.numpy as jnp

    specs = {
        "x": {"distribution": "uniform", "low": -3.0, "high": 7.0},
        "sigmain": {"distribution": "exponential", "rate": 1.0e-3},
    }

    def likelihood(p):
        return -0.5 * (p["x"] / p["sigmain"]) ** 2 - jnp.log(
            p["sigmain"] * jnp.sqrt(2 * jnp.pi)
        )

    model = adapter.build_jaxns_model(likelihood, specs)
    for point in ([0.2, 0.1], [0.5, 0.4], [0.8, 0.9]):
        unit = jnp.asarray(point)
        expected = {"x": -3 + 10 * unit[0], "sigmain": -jnp.log1p(-unit[1]) / 1e-3}
        actual = model.transform(unit)
        np.testing.assert_allclose(
            [actual[k] for k in specs], [expected[k] for k in specs]
        )
        assert float(model.forward(unit)) == pytest.approx(float(likelihood(expected)))
        physical_log_prior = -np.log(10) + np.log(1e-3) - 1e-3 * expected["sigmain"]
        assert float(model.log_prob_prior(unit)) == pytest.approx(
            float(physical_log_prior)
        )
        jacobian = jax.jacfwd(
            lambda u: jnp.array([model.transform(u)[name] for name in specs])
        )(unit)
        log_jacobian = jnp.linalg.slogdet(jacobian)[1]
        assert float(model.log_prob_prior(unit) + log_jacobian) == pytest.approx(
            0, abs=1e-12
        )


def test_real_jaxns_independent_gaussian_runs_match_known_evidence(
    jaxns_runtime, tmp_path
):
    import jax.numpy as jnp

    specs = {"x": {"distribution": "uniform", "low": -5.0, "high": 5.0}}

    def likelihood(p):
        return -0.5 * p["x"] ** 2 - 0.5 * jnp.log(2 * jnp.pi)

    expected_log_evidence = np.log(math.erf(5 / np.sqrt(2)) / 10)
    runs = []
    for seed in (17, 29):
        report, arrays = adapter.run_nested(
            likelihood, specs, seed, num_live_points=64, max_samples=2048, dlogz=0.03
        )
        assert report["converged"], report
        assert report["termination_reason"] == 4
        restored = adapter.validate_saved_results(report, arrays, specs)
        assert restored["converged"]
        assert restored["weighted_ess"] == report["weighted_ess"]
        assert abs(report["log_evidence"] - expected_log_evidence) < max(
            5 * report["log_evidence_error"], 0.1
        )
        assert abs(np.sum(arrays["weights"] * arrays["samples__x"])) < 0.4
        assert 1 <= report["weighted_ess"] <= report["num_samples"]
        np.testing.assert_allclose(np.sum(arrays["weights"]), 1)
        np.testing.assert_allclose(np.sum(np.exp(arrays["log_weights"])), 1)
        (tmp_path / f"seed-{seed}.json").write_text(json.dumps(report, allow_nan=False))
        np.savez(tmp_path / f"seed-{seed}.npz", **arrays)
        runs.append(arrays["samples__x"])
    assert not np.array_equal(*runs)


def test_real_numpyro_and_jaxns_share_normalized_joint_density(jaxns_runtime):
    numpyro = pytest.importorskip("numpyro")
    from numpyro import distributions
    from numpyro.infer.util import log_density
    import jax.numpy as jnp

    specs = {
        "x": {"distribution": "uniform", "low": -3.0, "high": 7.0},
        "sigmain": {"distribution": "exponential", "rate": 1e-3},
    }

    def likelihood(p):
        return -0.5 * (p["x"] / p["sigmain"]) ** 2 - jnp.log(
            p["sigmain"] * jnp.sqrt(2 * jnp.pi)
        )

    def numpyro_model():
        parameters = {
            "x": numpyro.sample("x", distributions.Uniform(-3.0, 7.0)),
            "sigmain": numpyro.sample("sigmain", distributions.Exponential(1e-3)),
        }
        numpyro.factor("observation", likelihood(parameters))

    model = adapter.build_jaxns_model(likelihood, specs)
    for point in ([0.2, 0.1], [0.5, 0.4], [0.8, 0.9]):
        unit = jnp.asarray(point)
        parameters = model.transform(unit)
        density, trace = log_density(numpyro_model, (), {}, parameters)
        assert float(model.forward(unit)) == pytest.approx(
            float(trace["observation"]["fn"].log_prob(trace["observation"]["value"])),
            abs=1e-12,
        )
        assert float(density) == pytest.approx(
            float(model.forward(unit) + model.log_prob_prior(unit)), abs=1e-12
        )


def test_real_jaxns_sample_budget_is_saved_as_unconverged(jaxns_runtime):
    import jax.numpy as jnp

    specs = {"x": {"distribution": "uniform", "low": -5.0, "high": 5.0}}
    report, arrays = adapter.run_nested(
        lambda p: -0.5 * p["x"] ** 2 - 0.5 * jnp.log(2 * jnp.pi),
        specs,
        17,
        num_live_points=16,
        max_samples=32,
        dlogz=1e-12,
    )
    assert report["status"] == "unconverged"
    assert report["termination_reason"] & 1
    assert arrays["log_weights"].size > 0
