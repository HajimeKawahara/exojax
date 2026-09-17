"""Run matched NUTS retrievals of a saved, validated CO/H2O mixture case.

Prepare and validate with ckd_mixture_validation.py first. Each run uses a
fresh process; summarization only reads saved evidence and never runs a model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

import _ckd_mixture_case as mixture
import compare_samplers as common


def _provenance(args):
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key != "handler"
    }
    return common._storage().collect_provenance(
        common.ROOT,
        [Path(__file__), Path(mixture.__file__), Path(common.__file__),
         common.ROOT / "tests/benchmark/benchmark_inference.py",
         common.ROOT / "tests/benchmark/diffgrid_nuts_storage.py"],
        settings,
    )


def _directory(output, run_id, method):
    return Path(output) / "runs" / common._storage().validate_run_id(run_id) / method


def _memory():
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
    except Exception as error:
        return {"available": False, "reason": str(error)}
    if stats is None:
        return {"available": False, "reason": "Backend does not report device memory."}
    return {"available": True, **{
        key: int(stats[key]) for key in ("bytes_in_use", "peak_bytes_in_use")
        if stats.get(key) is not None
    }}


def _validation(output, validation_id, method, context, environment, diagnostic=False):
    path = (Path(output) / "validations" /
            common._storage().validate_run_id(validation_id) / "validation.json")
    validation = common._storage().read_metadata(path)
    if validation.get("schema_version") != 1 or validation.get("status") != "completed":
        raise ValueError("Validation evidence must be completed with the current schema.")
    if validation["validation_id"] != validation_id:
        raise ValueError("Validation ID mismatch.")
    if validation["case_sha256"] != context["case_sha256"]:
        raise ValueError("Validation case digest differs from the selected case.")
    if validation["model_code_sha256"] != mixture.model_code_sha256():
        raise ValueError("Validation model code differs; validate this revision first.")
    if not validation["reference"]["passed"] or not validation["methods"][method]["gradient_passed"]:
        raise ValueError("Reference refinement and local gradient checks must pass even for diagnostic runs.")
    if not validation["methods"][method]["passed"] and not diagnostic:
        raise ValueError(f"Validation did not pass for {method}.")
    if validation["environment"] != environment:
        raise ValueError("Validation and sampling environments differ.")
    residual = path.with_name("spectra.npz")
    if common._storage().sha256(residual) != validation["residuals"]["sha256"]:
        raise ValueError("Validation residual archive digest mismatch.")
    return {
        "validation_id": validation_id, "sha256": common._storage().sha256(path),
        "model_code_sha256": validation["model_code_sha256"],
        "passed": bool(validation["methods"][method]["passed"]), "diagnostic": diagnostic,
    }


def _initialization(bounds, args):
    import jax
    import jax.numpy as jnp

    seed = args.seed if args.initialization_seed is None else args.initialization_seed
    fractions = np.random.default_rng(seed).uniform(
        0.05, 0.95, (args.num_chains, len(mixture.PARAMETER_ORDER))
    )
    positions = bounds[:, 0] + fractions * (bounds[:, 1] - bounds[:, 0])
    unconstrained = np.log(fractions / (1 - fractions))
    initial = {
        name: jnp.asarray(unconstrained[:, i] if args.num_chains > 1 else unconstrained[0, i])
        for i, name in enumerate(mixture.PARAMETER_ORDER)
    }
    warmup, sampling = jax.random.split(jax.random.PRNGKey(args.seed))
    if args.num_chains > 1:
        warmup, sampling = (jax.random.split(key, args.num_chains) for key in (warmup, sampling))
    record = {
        "policy": "prior", "seed": seed, "prior_fraction_range": [0.05, 0.95],
        "parameter_order": list(mixture.PARAMETER_ORDER), "positions": positions.tolist(),
        "warmup_keys": np.asarray(warmup).reshape(-1, 2).tolist(),
        "sampling_keys": np.asarray(sampling).reshape(-1, 2).tolist(),
    }
    return initial, warmup, sampling, record


def make_model(forward, bounds, sigma):
    """All methods induce the same independent uniform prior in named log coordinates."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(observation):
        theta = jnp.stack([
            numpyro.sample(name, dist.Uniform(bounds[i, 0], bounds[i, 1]))
            for i, name in enumerate(mixture.PARAMETER_ORDER)
        ])
        numpyro.sample("spectrum", dist.Normal(forward(theta), sigma), obs=observation)

    return model


def run(args):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    jax.config.update("jax_enable_x64", True)
    storage = common._storage()
    directory = _directory(args.output_dir, args.run_id, args.method)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "result.json"
    state = {"run_id": args.run_id, "method": args.method}
    with common._record_execution(path, state):
        state["provenance"] = _provenance(args)
        state["environment"] = mixture.runtime_environment()
        state["sampler_environment"] = common._environment()
        common._stage(path, state, "input_validation")
        context = mixture.load_context(args.output_dir)
        state["case_sha256"] = context["case_sha256"]
        state["case_metadata_sha256"] = storage.sha256(args.output_dir / "case.json")
        state["validation"] = _validation(
            args.output_dir, args.validation_id, args.method, context, state["environment"], args.diagnostic
        )
        state["quality_rules"] = dict(common._helper("benchmark_inference").DEFAULT_RULES)
        bounds = np.asarray(context["bounds"])
        initial, warmup_key, sampling_key, state["initialization"] = _initialization(bounds, args)
        state["settings"] = {
            "seed": args.seed, "num_chains": args.num_chains,
            "num_warmup": args.num_warmup, "num_samples": args.num_samples,
            "predictive_draws": args.predictive_draws,
            "diagnostic": args.diagnostic,
            "chain_method": "sequential", "dense_mass": True,
            "target_accept_probability": 0.9, "max_tree_depth": 10,
        }
        state["timings"] = {}
        state["device_memory"] = {"after_load": _memory()}
        forward = mixture.make_forward(context, args.method)
        model = make_model(forward, bounds, context["sigma"])
        sampler = MCMC(
            NUTS(model, dense_mass=True, target_accept_prob=0.9, max_tree_depth=10),
            num_warmup=args.num_warmup, num_samples=args.num_samples,
            num_chains=args.num_chains, chain_method="sequential", progress_bar=False,
        )
        observation = jnp.asarray(context["observed"])
        common._stage(path, state, "warmup")
        start = time.perf_counter()
        sampler.warmup(warmup_key, observation, init_params=initial)
        jax.block_until_ready(sampler.last_state)
        state["timings"]["compile_and_warmup_seconds"] = time.perf_counter() - start
        state["device_memory"]["after_warmup"] = _memory()
        common._stage(path, state, "sampling")
        start = time.perf_counter()
        sampler.run(sampling_key, observation, extra_fields=("num_steps", "accept_prob"))
        samples = sampler.get_samples(group_by_chain=True)
        extra = sampler.get_extra_fields(group_by_chain=True)
        jax.block_until_ready((samples, extra))
        state["timings"]["sampling_compile_and_run_seconds"] = time.perf_counter() - start
        state["device_memory"]["after_sampling"] = _memory()
        common._stage(path, state, "sample_save")
        samples, extra = jax.device_get((samples, extra))
        state["samples"] = storage.save_samples(
            directory / "samples.npz", samples, extra, mixture.PARAMETER_ORDER
        )
        common._stage(path, state, "diagnostics")
        state["posterior_inference"] = common._helper("benchmark_inference").posterior_diagnostics(
            samples, extra, state["quality_rules"]
        )
        common._stage(path, state, "posterior_predictive")
        indices = np.unique(np.linspace(0, args.num_samples - 1,
                                       min(args.predictive_draws, args.num_samples), dtype=int))
        positions = np.stack([samples[name][:, indices] for name in mixture.PARAMETER_ORDER], axis=-1)
        start = time.perf_counter()
        prediction = jax.jit(lambda xs: jax.lax.map(forward, xs))(
            jnp.asarray(positions.reshape(-1, positions.shape[-1]))
        )
        prediction = np.asarray(prediction).reshape(args.num_chains, len(indices), -1)
        state["timings"]["posterior_predictive_seconds"] = time.perf_counter() - start
        predictive_path = directory / "posterior_predictive.npz"
        storage.write_npz(predictive_path, prediction=prediction, draw_indices=indices,
                          observation=np.asarray(context["observed"]),
                          sigma=np.asarray(context["sigma"]), nu_bands=np.asarray(context["nu_bands"]))
        state["posterior_predictive"] = {
            "filename": predictive_path.name, "sha256": storage.sha256(predictive_path),
            "shape": list(prediction.shape), "finite": bool(np.isfinite(prediction).all()),
            "scope": "Noise-free model predictions at evenly spaced retained primary draws.",
        }
        state["measurement_scope"] = (
            "Synchronized compile+warmup and cold primary sampling, no supplementary draws; "
            "first-device memory high-water marks include earlier phases. Shared preparation "
            "and validation are excluded; launcher records whole-process time."
        )
    print(f"Saved {path}; posterior quality passed: {state['posterior_inference']['quality_passed']}")


def _load_run(output, run_id, method):
    storage = common._storage()
    directory = _directory(output, run_id, method)
    result = storage.read_metadata(directory / "result.json")
    if result.get("schema_version") != 2 or result.get("status") != "completed":
        raise ValueError("Run evidence must be completed with the current schema.")
    if result["run_id"] != run_id or result["method"] != method:
        raise ValueError("Run identity mismatch.")
    samples, extra = storage.load_samples(directory / "samples.npz", result["samples"])
    if result["samples"]["parameter_order"] != list(mixture.PARAMETER_ORDER):
        raise ValueError("Raw parameter order differs from the shared prior.")
    if result["samples"]["chain_shape"] != [result["settings"]["num_chains"], result["settings"]["num_samples"]]:
        raise ValueError("Raw sample dimensions differ from sampling controls.")
    if result["case_sha256"] != storage.sha256(Path(output) / "manifest.json"):
        raise ValueError("Prepared case manifest changed after sampling.")
    if result["case_metadata_sha256"] != storage.sha256(Path(output) / "case.json"):
        raise ValueError("Prepared case changed after sampling.")
    validation = Path(output) / "validations" / storage.validate_run_id(result["validation"]["validation_id"]) / "validation.json"
    if storage.sha256(validation) != result["validation"]["sha256"]:
        raise ValueError("Validation evidence changed after sampling.")
    evidence = storage.read_metadata(validation)
    if (evidence.get("schema_version") != 1 or evidence.get("status") != "completed" or
            evidence["case_sha256"] != result["case_sha256"] or
            evidence["model_code_sha256"] != result["validation"]["model_code_sha256"] or
            not evidence["reference"]["passed"] or not evidence["methods"][method]["gradient_passed"] or
            evidence["methods"][method]["passed"] != result["validation"]["passed"] or
            result["settings"]["diagnostic"] != result["validation"]["diagnostic"] or
            (not evidence["methods"][method]["passed"] and not result["settings"]["diagnostic"]) or
            evidence["environment"] != result["environment"]):
        raise ValueError("Validation is not compatible with the saved run.")
    if storage.sha256(validation.with_name("spectra.npz")) != evidence["residuals"]["sha256"]:
        raise ValueError("Validation residual digest mismatch.")
    prediction = directory / "posterior_predictive.npz"
    if storage.sha256(prediction) != result["posterior_predictive"]["sha256"]:
        raise ValueError("Posterior prediction digest mismatch.")
    with np.load(prediction, allow_pickle=False) as archive:
        values = archive["prediction"]
        indices = np.unique(np.linspace(
            0, result["settings"]["num_samples"] - 1,
            min(result["settings"]["predictive_draws"], result["settings"]["num_samples"]), dtype=int
        ))
        expected_shape = [result["settings"]["num_chains"], len(indices), len(archive["observation"])]
        if (list(values.shape) != expected_shape or expected_shape != result["posterior_predictive"]["shape"]
                or not np.isfinite(values).all() or not np.array_equal(archive["draw_indices"], indices)):
            raise ValueError("Invalid posterior predictions.")
        with np.load(Path(output) / "arrays.npz", allow_pickle=False) as case:
            for field, source in (("observation", "observed"), ("sigma", "sigma"), ("nu_bands", "nu_bands")):
                if not np.array_equal(archive[field], case[source]):
                    raise ValueError(f"Posterior prediction input differs from the case: {field}")
    result["posterior_inference"] = common._helper("benchmark_inference").posterior_diagnostics(
        samples, extra, result["quality_rules"]
    )
    result["result_sha256"] = storage.sha256(directory / "result.json")
    elapsed = result["timings"]["sampling_compile_and_run_seconds"]
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Sampling duration must be finite and positive.")
    result["efficiency"] = {
        name: {key + "_per_cold_sampling_second": record[key] / elapsed if record[key] is not None else None
               for key in ("ess_bulk", "ess_tail")}
        for name, record in result["posterior_inference"]["per_parameter"].items()
    }
    return result


def summarize(args):
    storage = common._storage()
    mixture.verify_case(args.output_dir)
    run_ids = [args.run_id, *args.repeat_run_id]
    if len(set(run_ids)) != len(run_ids) or len(set(args.methods)) != len(args.methods):
        raise ValueError("Run IDs and method selections must be unique.")
    if len(args.methods) < 2:
        raise ValueError("Select at least two methods.")
    directory = Path(args.output_dir) / "runs" / storage.validate_run_id(args.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    state = {"run_ids": run_ids, "methods": args.methods}
    with common._record_execution(directory / "comparison.json", state):
        pairs = [{m: _load_run(args.output_dir, rid, m) for m in args.methods} for rid in run_ids]
        reference = pairs[0][args.methods[0]]
        reasons, seeds, initialization_seeds, keys = [], set(), set(), set()
        if len(pairs) < 2:
            reasons.append("At least two independent matched pairs are required.")
        for pair in pairs:
            first = pair[args.methods[0]]
            current_keys = {tuple(key) for group in ("warmup_keys", "sampling_keys")
                            for key in first["initialization"][group]}
            if len(current_keys) != 2 * first["settings"]["num_chains"]:
                raise ValueError("Warmup and sampling keys must be unique for every chain.")
            if (first["settings"]["seed"] in seeds or first["initialization"]["seed"] in initialization_seeds
                    or keys.intersection(current_keys)):
                raise ValueError("Repeated runs must use independent seeds and initial states.")
            seeds.add(first["settings"]["seed"])
            initialization_seeds.add(first["initialization"]["seed"])
            keys.update(current_keys)
            for method, result in pair.items():
                if result["settings"] != first["settings"] or result["initialization"] != first["initialization"]:
                    raise ValueError("Matched methods must use identical controls and starting states.")
                normalized, baseline = copy.deepcopy(result["settings"]), copy.deepcopy(reference["settings"])
                normalized.pop("seed"); baseline.pop("seed")
                if normalized != baseline:
                    raise ValueError("Repeat sampling controls differ.")
                for field in ("case_sha256", "case_metadata_sha256", "environment", "sampler_environment",
                              "quality_rules"):
                    if result[field] != reference[field]:
                        raise ValueError(f"Comparison conditions differ: {field}")
                for field in ("validation_id", "sha256", "model_code_sha256", "diagnostic"):
                    if result["validation"][field] != reference["validation"][field]:
                        raise ValueError(f"Validation conditions differ: {field}")
                for field in ("code_sha256", "dependencies", "environment"):
                    if result["provenance"][field] != reference["provenance"][field]:
                        raise ValueError(f"Execution provenance differs: {field}")
                if result["quality_rules"] != common._helper("benchmark_inference").DEFAULT_RULES:
                    reasons.append("The predefined posterior quality rules were changed.")
                if not result["posterior_inference"]["quality_passed"]:
                    reasons.append(f"{result['run_id']}/{method}: posterior convergence failed.")
                if result["settings"]["diagnostic"] or not result["validation"]["passed"]:
                    reasons.append(f"{result['run_id']}/{method}: diagnostic retrieval is not a scientific performance comparison.")
                if result["settings"]["num_warmup"] < 500 or result["settings"]["num_samples"] < 1000:
                    reasons.append(f"{result['run_id']}/{method}: requires >=500 warmup and >=1000 draws.")
        state["quality"] = {"eligible": not reasons, "reasons": reasons}
        state["repetitions"] = pairs
        state["posterior_comparisons"] = [{
            method: common._helper("benchmark_inference").posterior_difference(
                pair[args.methods[0]]["posterior_inference"], pair[method]["posterior_inference"]
            ) for method in args.methods[1:]
        } for pair in pairs]
        state["cold_sampling_speedups"] = {
            method: {
                "ratios": [pair[args.methods[0]]["timings"]["sampling_compile_and_run_seconds"] /
                           pair[method]["timings"]["sampling_compile_and_run_seconds"] for pair in pairs],
                "scientific_median": float(np.median([
                    pair[args.methods[0]]["timings"]["sampling_compile_and_run_seconds"] /
                    pair[method]["timings"]["sampling_compile_and_run_seconds"] for pair in pairs
                ])) if not reasons else None,
            } for method in args.methods[1:]
        }
        state["scope"] = "Fixed small molecular sample, atmosphere, noise realization and backend; no coverage or broad-band accuracy claim."
    print(json.dumps(state["quality"], indent=2))
    print(f"Saved {directory / 'comparison.json'}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sampling = commands.add_parser("run")
    sampling.add_argument("--output-dir", type=Path, required=True)
    sampling.add_argument("--method", choices=mixture.METHODS, required=True)
    sampling.add_argument("--run-id", type=common._run_id, required=True)
    sampling.add_argument("--validation-id", type=common._run_id, required=True)
    sampling.add_argument("--seed", type=common._sampler_seed, default=0)
    sampling.add_argument("--initialization-seed", type=common._sampler_seed)
    sampling.add_argument("--num-chains", type=common._positive_int, default=4)
    sampling.add_argument("--num-warmup", type=common._positive_int, default=500)
    sampling.add_argument("--num-samples", type=common._positive_int, default=1000)
    sampling.add_argument("--predictive-draws", type=common._positive_int, default=100)
    sampling.add_argument("--diagnostic", action="store_true", help=(
        "Study posterior bias despite failed spectral accuracy; reference refinement and local "
        "gradient checks must still pass. Always excluded from scientific performance summaries."
    ))
    sampling.set_defaults(handler=run)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--run-id", type=common._run_id, required=True)
    summary.add_argument("--repeat-run-id", type=common._run_id, action="append", default=[])
    summary.add_argument("--methods", choices=mixture.METHODS, nargs="+", default=["lbl", "rorr"])
    summary.set_defaults(handler=summarize)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.handler(arguments)
