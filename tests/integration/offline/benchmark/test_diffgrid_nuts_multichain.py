"""Offline contracts for multiple-chain benchmark runs and comparisons."""

import copy
import csv
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tests" / "benchmark"))
    return importlib.import_module("diffgrid_nuts_benchmark")


@pytest.fixture
def tiny_case(benchmark, monkeypatch, tmp_path):
    """Keep the real sampler/model/storage around a six-coordinate linear spectrum."""
    import jax.numpy as jnp

    storage = importlib.import_module("diffgrid_nuts_storage")
    paths = benchmark._case_paths(tmp_path)
    configuration = benchmark.CaseConfig(
        number_of_wavenumbers=6,
        number_of_observed_wavenumbers=6,
        number_of_layers=2,
    )
    nu_grid = np.linspace(1000.0, 1001.0, 6)
    truth = benchmark.TRUTH
    observed = np.array(
        [
            truth["radius"],
            truth["radial_velocity"] / 10.0,
            truth["methane_mass_mixing_ratio"] / 0.01,
            truth["temperature_at_1bar"] / 1000.0,
            truth["temperature_index"] / 0.1,
            truth["vsini"] / 20.0,
        ]
    )
    storage.write_npz(
        paths["case"],
        observed_flux=observed,
        teacher_flux=observed,
        nu_grid=nu_grid,
        nu_data=nu_grid,
        model_resolution=np.array(1000.0),
    )
    for name in ("premodit", "premodit_metadata", "diffgrid", "diffgrid_metadata"):
        paths[name].write_bytes(name.encode())
    cia = tmp_path / "synthetic.cia"
    cia.write_bytes(b"synthetic CIA")
    metadata = {
        "schema_version": 2,
        "status": "completed",
        "config": asdict(configuration),
        "truth": truth,
        "prior_bounds": benchmark.PRIOR_BOUNDS,
        "artifacts": {
            f"{name}_sha256": storage.sha256(path)
            for name, path in paths.items()
            if name != "prepare"
        },
        "inputs": {"cia_path": str(cia), "cia_sha256": storage.sha256(cia)},
        "timings": {"diffgrid_build_seconds": 1.0},
        "diffgrid": {
            "table_payload_bytes": 64,
            "maximum_interpolation_error_in_noise": 0.001,
        },
    }
    storage.write_json(paths["prepare"], metadata)
    art = SimpleNamespace(
        pressure=jnp.array([0.1, 1.0]),
        powerlaw_temperature=lambda temperature, index: jnp.array([temperature, index]),
    )
    context = {"art": art, "nu_data": jnp.asarray(nu_grid)}

    def forward(temperature, methane, radius, radial_velocity, vsini):
        return jnp.array(
            [
                radius,
                radial_velocity / 10.0,
                methane / 0.01,
                temperature[0] / 1000.0,
                temperature[1] / 0.1,
                vsini / 20.0,
            ]
        )

    monkeypatch.setattr(benchmark, "_load_scientific_runtime", lambda: None)
    monkeypatch.setattr(benchmark, "_block_opacity", lambda opacity: None)
    monkeypatch.setattr(
        benchmark,
        "_load_opacity",
        lambda method, *args: SimpleNamespace(
            method=method,
            nu_grid=nu_grid,
            check_pressure_grid=lambda pressure: None,
        ),
    )
    monkeypatch.setattr(benchmark, "_forward_context", lambda *args: context)
    monkeypatch.setattr(benchmark, "_make_forward_model", lambda *args: forward)
    monkeypatch.setattr(
        benchmark, "_physical_metadata", lambda *args: {"model": "synthetic linear"}
    )
    monkeypatch.setattr(
        benchmark,
        "_provenance",
        lambda *args: {
            "code_sha256": "synthetic-code",
            "git": {"commit": "synthetic"},
            "environment": {"JAX_PLATFORMS": "cpu"},
            "dependencies": {"numpy": "test"},
        },
    )
    monkeypatch.setattr(benchmark, "_plot_comparison", lambda *args: None)
    return paths, metadata, storage


def test_chain_cli_keeps_legacy_defaults_and_records_explicit_selection(benchmark):
    parser = benchmark._parser()
    legacy = parser.parse_args(["run", "--method", "premodit"])
    assert legacy.num_chains == 1
    assert legacy.chain_method == "sequential"
    assert legacy.initialization == "truth"
    assert legacy.initialization_seed is None
    assert legacy.measure_steady_sampling is False
    explicit = parser.parse_args(
        [
            "run",
            "--method",
            "diffgrid",
            "--num-chains",
            "4",
            "--chain-method",
            "vectorized",
            "--initialization",
            "prior",
            "--initialization-seed",
            "17",
        ]
    )
    assert (explicit.num_chains, explicit.chain_method) == (4, "vectorized")
    assert (explicit.initialization, explicit.initialization_seed) == ("prior", 17)
    comparison = parser.parse_args(
        ["summarize", "--run-id", "first", "--repeat-run-id", "second"]
    )
    assert comparison.run_id == "first"
    assert comparison.repeat_run_id == ["second"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--method", "premodit", "--num-chains", "0"],
        ["run", "--method", "premodit", "--chain-method", "automatic"],
        ["run", "--method", "premodit", "--initialization", "unknown"],
        ["summarize", "--repeat-run-id", "../other"],
    ],
)
def test_invalid_chain_cli_is_rejected(benchmark, arguments):
    with pytest.raises(SystemExit):
        benchmark._parser().parse_args(arguments)


def test_legacy_truth_initialization_preserves_exact_random_key_path(benchmark):
    import jax

    args = benchmark._parser().parse_args(
        ["run", "--method", "premodit", "--seed", "19"]
    )
    metadata, initial, warmup, sampling = benchmark._chain_initialization(
        args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
    )
    expected_warmup, expected_sampling = jax.random.split(jax.random.PRNGKey(19))
    assert initial is None
    assert metadata["policy"] == "truth"
    assert metadata["physical_values"] == [benchmark.TRUTH]
    np.testing.assert_array_equal(warmup, expected_warmup)
    np.testing.assert_array_equal(sampling, expected_sampling)


def test_prior_initialization_is_shared_dispersed_and_reproducible(benchmark):
    args = benchmark._parser().parse_args(
        [
            "run",
            "--method",
            "premodit",
            "--num-chains",
            "4",
            "--initialization",
            "prior",
            "--seed",
            "29",
            "--initialization-seed",
            "41",
        ]
    )
    metadata, initial, warmup, sampling = benchmark._chain_initialization(
        args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
    )
    other_args = copy.copy(args)
    other_args.method = "diffgrid"
    other = benchmark._chain_initialization(
        other_args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
    )
    assert metadata == other[0]
    assert metadata["policy"] == "prior"
    assert metadata["seed"] == 41
    assert np.asarray(warmup).shape == np.asarray(sampling).shape == (4, 2)
    assert len({tuple(key) for key in np.concatenate([warmup, sampling])}) == 8
    for name, (low, high) in benchmark.PRIOR_BOUNDS.items():
        physical = np.array([point[name] for point in metadata["physical_values"]])
        assert np.all((low < physical) & (physical < high))
        assert len(set(physical)) == 4
        restored = low + (high - low) / (1.0 + np.exp(-np.asarray(initial[name])))
        np.testing.assert_allclose(restored, physical, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(initial[name], other[1][name])
    other_args.seed += 1
    changed = benchmark._chain_initialization(
        other_args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
    )
    assert metadata["physical_values"] == changed[0]["physical_values"]
    assert not np.array_equal(warmup, changed[2])
    other_args.initialization_seed += 1
    changed_initial = benchmark._chain_initialization(
        other_args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
    )
    assert metadata["physical_values"] != changed_initial[0]["physical_values"]


def test_parallel_chain_device_shortage_is_saved_as_failure(
    benchmark, tiny_case, monkeypatch, tmp_path
):
    pytest.importorskip("numpyro")
    monkeypatch.setattr(benchmark.jax, "local_device_count", lambda: 1)
    args = benchmark._parser().parse_args(
        [
            "run",
            "--method",
            "premodit",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "parallel",
            "--num-chains",
            "2",
            "--chain-method",
            "parallel",
            "--num-warmup",
            "2",
            "--num-samples",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="device|parallel"):
        benchmark.run_method(args)
    result_path = tiny_case[2].result_paths(tmp_path, "premodit", "parallel")["result"]
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert "device" in result["failure"]["message"].lower()


@pytest.mark.parametrize("chain_method", ["sequential", "vectorized"])
def test_real_cpu_two_chain_smoke_persists_samples_and_recomputes_diagnostics(
    benchmark, tiny_case, tmp_path, chain_method
):
    pytest.importorskip("numpyro")
    paths, _, storage = tiny_case
    args = benchmark._parser().parse_args(
        [
            "run",
            "--method",
            "premodit",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "cpu-smoke",
            "--num-chains",
            "2",
            "--chain-method",
            chain_method,
            "--initialization",
            "prior",
            "--num-warmup",
            "20",
            "--num-samples",
            "8",
            "--gradient-repetitions",
            "1",
            "--predictive-draws",
            "3",
            "--measure-steady-sampling",
        ]
    )
    benchmark.run_method(args)
    saved = storage.result_paths(tmp_path, "premodit", "cpu-smoke")
    result = json.loads(saved["result"].read_text())
    assert result["status"] == "completed"
    assert result["run"]["num_chains"] == 2
    assert result["run"]["chain_method"] == chain_method
    assert result["run"]["initialization"]["policy"] == "prior"
    samples, extra = storage.load_samples(saved["samples"], result["samples"])
    assert set(samples) == set(benchmark.TRUTH)
    assert all(value.shape == (2, 8) for value in samples.values())
    assert all(np.isfinite(value).all() for value in samples.values())
    assert extra["num_steps"].shape == (2, 8)
    assert not np.array_equal(samples["radius"][0], samples["radius"][1])
    assert result["steady_sampling"]["available"]
    assert result["steady_sampling"]["seconds"] > 0.0
    steady, _ = storage.load_samples(
        saved["result"].with_name(result["steady_samples"]["filename"]),
        result["steady_samples"],
    )
    assert steady["radius"].shape == samples["radius"].shape
    assert not np.array_equal(steady["radius"], samples["radius"])
    prediction = result["posterior_predictive"]
    assert prediction["available"]
    prediction_path = saved["result"].with_name(prediction["filename"])
    assert storage.sha256(prediction_path) == prediction["sha256"]
    with np.load(prediction_path, allow_pickle=False) as archive:
        assert archive["prediction"].shape == (2, 3, 6)
        assert archive["replicated_observation"].shape == (2, 3, 6)
        assert not np.array_equal(
            archive["prediction"], archive["replicated_observation"]
        )
        np.testing.assert_array_equal(
            archive["nu_data"], np.linspace(1000.0, 1001.0, 6)
        )
    assert result["posterior_inference"]["quality_passed"] is False
    assert result["forward_benchmark"]["compile_seconds"] >= 0.0
    assert result["forward_benchmark"]["median_evaluation_seconds"] > 0.0
    result["diagnostics"]["total_num_steps"] = -1000
    storage.write_json(saved["result"], result)
    restored = benchmark._read_result(
        tmp_path, "premodit", "cpu-smoke", storage.sha256(paths["prepare"])
    )
    assert restored["diagnostics"]["total_num_steps"] == int(extra["num_steps"].sum())
    assert restored["diagnostics_source"] == "recomputed from saved chains"
    saved["samples"].write_bytes(saved["samples"].read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="digest"):
        benchmark._read_result(
            tmp_path, "premodit", "cpu-smoke", storage.sha256(paths["prepare"])
        )


@pytest.fixture
def saved_pairs(benchmark, tiny_case, monkeypatch, tmp_path):
    """Exercise comparison control flow with deterministic diagnostic outcomes."""
    inference = importlib.import_module("benchmark_inference")
    paths, metadata, storage = tiny_case
    prepare_digest = storage.sha256(paths["prepare"])

    def diagnostics(samples, extra, rules=None):
        passed = not bool(np.any(extra["diverging"]))
        per_parameter = {
            name: {
                "mean": float(np.mean(values)),
                "quantiles": {
                    str(q): float(np.quantile(values, q)) for q in (0.05, 0.5, 0.95)
                },
                "mcse_mean": 0.01,
                "mcse_quantiles": {str(q): 0.02 for q in (0.05, 0.5, 0.95)},
                "rhat_rank": 1.001,
                "ess_bulk": 1000.0,
                "ess_tail": 900.0,
                "quality_passed": passed,
                "failure_reasons": [] if passed else ["Divergences"],
            }
            for name, values in samples.items()
        }
        shape = next(iter(samples.values())).shape
        return {
            "status": "completed",
            "quality_passed": passed,
            "failure_reasons": [] if passed else ["Divergences"],
            "num_chains": shape[0],
            "num_draws": shape[1],
            "rules": dict(inference.DEFAULT_RULES),
            "implementation": {"name": "synthetic", "version": "1", "available": True},
            "definitions": {},
            "per_parameter": per_parameter,
            "summary": {
                "max_rhat_rank": 1.001,
                "min_ess_bulk": 1000.0,
                "min_ess_tail": 900.0,
                "divergences": int(np.sum(extra["diverging"])),
            },
        }

    monkeypatch.setattr(inference, "posterior_diagnostics", diagnostics)
    if hasattr(benchmark, "posterior_diagnostics"):
        monkeypatch.setattr(benchmark, "posterior_diagnostics", diagnostics)
    monkeypatch.setattr(
        benchmark,
        "_validate_result_evidence",
        lambda output, result, *args: result.update(
            accuracy_validation={"status": "passed", "passed": True}
        ),
    )
    monkeypatch.setattr(
        benchmark, "_minimum_effective_sample_size", lambda samples: 1000.0
    )
    for run_id, seed in (("first", 11), ("second", 22)):
        args = benchmark._parser().parse_args(
            [
                "run",
                "--method",
                "premodit",
                "--num-chains",
                "4",
                "--initialization",
                "prior",
                "--seed",
                str(seed),
            ]
        )
        initialization, _, _, _ = benchmark._chain_initialization(
            args, benchmark.TRUTH, benchmark.PRIOR_BOUNDS
        )
        for method in ("premodit", "diffgrid"):
            saved = storage.reserve_result(tmp_path, method, run_id)
            rng = np.random.default_rng(seed)
            samples = {
                name: rng.normal(value, max(abs(value), 0.001) * 0.01, (4, 1000))
                for name, value in benchmark.TRUTH.items()
            }
            extra = {
                "num_steps": np.ones((4, 1000), dtype=np.int32),
                "diverging": np.zeros((4, 1000), dtype=bool),
                "accept_prob": np.full((4, 1000), 0.95),
            }
            prediction = np.stack(
                [samples[name][:, [0, 999]] for name in benchmark.TRUTH], axis=-1
            )
            prediction_path = saved["result"].with_name("posterior_predictive.npz")
            storage.write_npz(
                prediction_path,
                prediction=prediction,
                replicated_observation=prediction + 0.01,
                draw_indices=np.array([0, 999]),
                nu_data=np.linspace(1000.0, 1001.0, 6),
            )
            result = {
                "schema_version": 2,
                "status": "completed",
                "method": method,
                "run_id": run_id,
                "case_sha256": metadata["artifacts"]["case_sha256"],
                "prepare_sha256": prepare_digest,
                "run": {
                    "num_chains": 4,
                    "num_samples": 1000,
                    "num_warmup": 500,
                    "seed": seed,
                    "chain_method": "sequential",
                    "initialization": initialization,
                },
                "environment": benchmark._environment(),
                "physics": {"model": "synthetic linear"},
                "provenance": benchmark._provenance(None, None),
                "timings": {
                    "opacity_load_seconds": 0.1,
                    "model_setup_seconds": 0.1,
                    "compile_and_warmup_seconds": 1.0,
                    "sampling_compile_and_run_seconds": 2.0
                    if method == "premodit"
                    else 1.0,
                    "sampling_seconds_per_sample": 0.001,
                    "cold_milliseconds_per_leapfrog_step": 0.001,
                },
                "potential_gradient_benchmark": {"median_evaluation_seconds": 0.01},
                "diagnostics": {},
                "posterior_inference": diagnostics(samples, extra),
                "quality_rules": dict(inference.DEFAULT_RULES),
                "posterior_predictive": {
                    "available": True,
                    "filename": prediction_path.name,
                    "sha256": storage.sha256(prediction_path),
                    "shape": list(prediction.shape),
                    "draw_indices": [0, 999],
                    "prediction": inference.predictive_summary(prediction),
                    "replicated_observation": inference.predictive_summary(
                        prediction + 0.01
                    ),
                },
                "device_memory": {},
                "host_peak_rss_bytes": None,
                "samples": storage.save_samples(
                    saved["samples"], samples, extra, list(benchmark.TRUTH)
                ),
            }
            storage.write_json(saved["result"], result)
    return storage


def test_repeat_cli_rejects_ambiguous_selection(benchmark, saved_pairs, tmp_path):
    for arguments in (
        ["--repeat-run-id", "second"],
        ["--run-id", "first", "--repeat-run-id", "first"],
        ["--run-id", "first", "--repeat-run-id", "second", "--repeat-run-id", "second"],
        [
            "--run-id",
            "first",
            "--repeat-run-id",
            "second",
            "--compare-run-id",
            "third",
            "--method",
            "premodit",
        ],
    ):
        args = benchmark._parser().parse_args(
            ["summarize", "--output-dir", str(tmp_path), *arguments]
        )
        with pytest.raises(ValueError, match="run|repeat|comparison|requires"):
            benchmark.summarize_results(args)


@pytest.mark.parametrize(
    "mismatch",
    [
        "device",
        "seed",
        "initialization",
        "initialization_policy",
        "initialization_range",
        "quality_rules",
    ],
)
def test_repeat_comparison_rejects_mismatched_conditions_and_keeps_failure(
    benchmark, saved_pairs, tmp_path, mismatch
):
    storage = saved_pairs
    selected = (
        [("second", method) for method in ("premodit", "diffgrid")]
        if mismatch
        in ("device", "seed", "initialization_policy", "initialization_range")
        else [("first", "diffgrid")]
    )
    first = json.loads(
        storage.result_paths(tmp_path, "premodit", "first")["result"].read_text()
    )
    for run_id, method in selected:
        path = storage.result_paths(tmp_path, method, run_id)["result"]
        result = json.loads(path.read_text())
        if mismatch == "device":
            result["environment"]["device_kind"] = "different device"
        elif mismatch == "seed":
            result["run"]["seed"] = first["run"]["seed"]
            result["run"]["initialization"] = first["run"]["initialization"]
        elif mismatch == "initialization":
            result["run"]["initialization"]["physical_values"][0]["radius"] += 0.001
        elif mismatch == "initialization_policy":
            result["run"]["initialization"]["policy"] = "truth"
        elif mismatch == "initialization_range":
            result["run"]["initialization"]["prior_fraction_range"] = [0.1, 0.9]
        else:
            result["quality_rules"]["max_rhat"] = 1.2
        storage.write_json(path, result)
    args = benchmark._parser().parse_args(
        [
            "summarize",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "first",
            "--repeat-run-id",
            "second",
        ]
    )
    summary_path = tmp_path / "runs/first/comparison.json"
    storage.write_json(
        summary_path, {"status": "completed", "quality": {"eligible": True}}
    )
    with pytest.raises(ValueError):
        benchmark.summarize_results(args)
    failed = json.loads((tmp_path / "runs/first/comparison_failed.json").read_text())
    assert failed["status"] == "failed"
    assert failed["reason"]
    assert failed["repeat_run_ids"] == ["second"]
    assert json.loads(summary_path.read_text()) == failed


@pytest.mark.parametrize("repeat", [False, True])
def test_scientific_cost_requires_independent_successful_run_pairs(
    benchmark, saved_pairs, tmp_path, repeat
):
    arguments = ["summarize", "--output-dir", str(tmp_path), "--run-id", "first"]
    if repeat:
        arguments += ["--repeat-run-id", "second"]
    benchmark.summarize_results(benchmark._parser().parse_args(arguments))
    comparison = json.loads((tmp_path / "runs/first/comparison.json").read_text())
    assert comparison["status"] == "completed"
    assert comparison["quality"]["eligible"] is repeat
    assert comparison["scientific_sampling_speedup_premodit_over_diffgrid"] == (
        2.0 if repeat else None
    )
    assert comparison["sampling_speedup_premodit_over_diffgrid"] == 2.0
    with (tmp_path / "runs/first/comparison.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    selected = ["first", "second"] if repeat else ["first"]
    assert len(costs) == 2 * len(selected)
    assert {(row["run_id"], row["method"]) for row in costs} == {
        (run_id, method) for run_id in selected for method in ("premodit", "diffgrid")
    }
    if repeat:
        assert [record["run_id"] for record in comparison["repetitions"]] == [
            "first",
            "second",
        ]
    else:
        assert any(
            "independent" in reason for reason in comparison["quality"]["reasons"]
        )


def test_failed_saved_chain_cannot_be_rescued_by_stale_quality_metadata(
    benchmark, saved_pairs, tmp_path
):
    storage = saved_pairs
    saved = storage.result_paths(tmp_path, "diffgrid", "second")
    result = json.loads(saved["result"].read_text())
    assert result["posterior_inference"]["quality_passed"] is True
    samples, extra = storage.load_samples(saved["samples"], result["samples"])
    extra["diverging"][0, 0] = True
    result["samples"] = storage.save_samples(
        saved["samples"], samples, extra, result["samples"]["parameter_order"]
    )
    storage.write_json(saved["result"], result)
    arguments = [
        "summarize",
        "--output-dir",
        str(tmp_path),
        "--run-id",
        "first",
        "--repeat-run-id",
        "second",
    ]
    benchmark.summarize_results(benchmark._parser().parse_args(arguments))
    comparison = json.loads((tmp_path / "runs/first/comparison.json").read_text())
    assert comparison["status"] == "completed"
    assert comparison["quality"]["eligible"] is False
    assert comparison["scientific_sampling_speedup_premodit_over_diffgrid"] is None
    failed = comparison["repetitions"][1]["methods"]["diffgrid"]
    assert failed["posterior_inference"]["quality_passed"] is False
    assert failed["posterior_inference"]["summary"]["divergences"] == 1
    assert failed["diagnostics_source"] == "recomputed from saved chains"


@pytest.mark.parametrize("field", ["nu_data", "draw_indices", "replicated_observation"])
def test_predictive_archive_contract_rejects_inconsistent_contents(
    benchmark, saved_pairs, tmp_path, field
):
    storage = saved_pairs
    saved = storage.result_paths(tmp_path, "diffgrid", "first")
    result = json.loads(saved["result"].read_text())
    manifest = result["posterior_predictive"]
    path = saved["result"].with_name(manifest["filename"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if field == "nu_data":
        arrays[field] = arrays[field][::-1]
    elif field == "draw_indices":
        arrays[field] = np.array([0, 1000])
        manifest["draw_indices"] = [0, 1000]
    else:
        arrays[field] = arrays[field][..., :1]
    storage.write_npz(path, **arrays)
    manifest["sha256"] = storage.sha256(path)
    storage.write_json(saved["result"], result)
    with pytest.raises(
        ValueError, match="predictive|Predictive|draw|observation|wavenumber"
    ):
        benchmark._read_result(
            tmp_path,
            "diffgrid",
            "first",
            storage.sha256(benchmark._case_paths(tmp_path)["prepare"]),
        )


def test_whole_process_time_uses_only_the_selected_run_log(
    benchmark, saved_pairs, tmp_path
):
    storage = saved_pairs
    prepare_digest = storage.sha256(benchmark._case_paths(tmp_path)["prepare"])
    missing = benchmark._read_result(tmp_path, "premodit", "first", prepare_digest)[
        "whole_process_time"
    ]
    assert missing["available"] is False
    assert all(missing[field] is None for field in ("real", "user", "sys"))
    assert missing["reason"]
    directory = tmp_path / "process_times"
    directory.mkdir()
    (directory / "second-premodit.txt").write_text("real 9.0\nuser 8.0\nsys 7.0\n")
    assert (
        benchmark._read_process_time(tmp_path, "premodit", "first")["available"]
        is False
    )
    path = directory / "first-premodit.txt"
    path.write_text("real 0.25\nuser 1.50\nsys 0.10\n")
    measured = benchmark._read_result(tmp_path, "premodit", "first", prepare_digest)[
        "whole_process_time"
    ]
    assert measured["available"] is True
    assert [measured[field] for field in ("real", "user", "sys")] == [0.25, 1.5, 0.1]
    assert measured["filename"] == str(path)
    assert measured["sha256"] == storage.sha256(path)


@pytest.mark.parametrize(
    "contents",
    [
        "real 0.25\nuser 1.50\n",
        "real nan\nuser 1.50\nsys 0.10\n",
        "real -0.25\nuser 1.50\nsys 0.10\n",
        "real invalid\nuser 1.50\nsys 0.10\n",
    ],
)
def test_invalid_whole_process_time_is_rejected(benchmark, tmp_path, contents):
    directory = tmp_path / "process_times"
    directory.mkdir()
    (directory / "first-premodit.txt").write_text(contents)
    with pytest.raises(ValueError):
        benchmark._read_process_time(tmp_path, "premodit", "first")
