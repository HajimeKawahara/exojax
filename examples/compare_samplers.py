"""Compare the CO tutorials on one saved observation: prepare, run, summarize.

Run from a source checkout. NumPyro, ArviZ, and JAXNS remain optional, and each
sampler runs in a fresh process. NUTS chain ESS and nested weighted ESS have
different meanings; this example does not rank their evidence capabilities.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time

import numpy as np

from _co_retrieval import (
    CaseConfig,
    PRIOR_SPECS,
    UNITS,
    make_forward,
    make_numpyro_model,
    mock_truth,
    normalized_log_prior,
    physical_log_likelihood,
    predict,
    unit_to_physical,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/co_sampler_comparison"


@lru_cache(None)
def _helper(name):
    """Reuse the existing P0 storage/diagnostics without importing its CLI."""
    path = ROOT / "tests/benchmark" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_co_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _storage():
    return _helper("diffgrid_nuts_storage")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment():
    import jax

    return {
        "python": platform.python_version(),
        "versions": {
            name: _version(name)
            for name in (
                "exojax",
                "numpy",
                "jax",
                "jaxlib",
                "numpyro",
                "arviz",
                "jaxns",
                "tfp-nightly",
            )
        },
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()],
        "device_kind": jax.devices()[0].device_kind,
        "backend": jax.default_backend(),
    }


def _provenance(args, inputs):
    settings = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
        if name != "handler"
    }
    files = [
        Path(__file__),
        Path(__file__).with_name("_co_retrieval.py"),
        Path(__file__).with_name("_compare_samplers_jaxns.py"),
        ROOT / "tests/benchmark/diffgrid_nuts_storage.py",
        ROOT / "tests/benchmark/benchmark_inference.py",
        *inputs,
    ]
    provenance = _storage().collect_provenance(ROOT, files, settings)
    provenance["dependencies"].update(
        {name: _version(name) for name in ("jaxns", "tfp-nightly")}
    )
    return provenance


@contextmanager
def _record_execution(path, state):
    storage = _storage()
    state.update(
        schema_version=storage.SCHEMA_VERSION, status="partial", stage="initializing"
    )
    storage.write_json(path, state)
    started = time.perf_counter()
    try:
        yield
    except BaseException as error:
        state.update(
            status="failed",
            failure={
                "stage": state["stage"],
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    else:
        state.update(status="completed", stage="completed")
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        storage.write_json(path, state)


def _stage(path, state, name):
    state["stage"] = name
    _storage().write_json(path, state)


def _context(nu_grid, nu_obs, opa, cia_path, config):
    from exojax.database.cia.api import CdbCIA
    from exojax.opacity import OpaCIA
    from exojax.postproc.specop import SopInstProfile, SopRotation
    from exojax.rt import ArtEmisPure
    from exojax.utils.instfunc import resolution_to_gaussian_std

    art = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=config.pressure_top,
        pressure_btm=config.pressure_bottom,
        nlayer=config.number_of_layers,
        rtsolver="ibased",
        nstream=8,
    )
    art.change_temperature_range(config.temperature_min, config.temperature_max)
    return {
        "nu_grid": np.asarray(nu_grid),
        "nu_obs": np.asarray(nu_obs),
        "opa": opa,
        "art": art,
        "molmass": float(opa.molmass),
        "opacia": OpaCIA(CdbCIA(str(cia_path), nurange=nu_grid), nu_grid=nu_grid),
        "sop_rot": SopRotation(nu_grid, vsini_max=config.maximum_vsini),
        "sop_inst": SopInstProfile(nu_grid, vrmax=config.maximum_instrument_velocity),
        "beta_inst": resolution_to_gaussian_std(config.instrument_resolution),
        "vmrH2": config.hydrogen_volume_mixing_ratio,
        "mmw": config.mean_molecular_weight,
        "cia_path": Path(cia_path).resolve(),
    }


def _prepare_context(args, config):
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit
    from exojax.utils.grids import wavenumber_grid

    nu, _, _ = wavenumber_grid(
        config.wavelength_min,
        config.wavelength_max,
        config.number_of_wavenumbers,
        unit="AA",
        xsmode="premodit",
    )
    observed = nu[:: config.observation_stride]
    if config.observation_trim:
        observed = observed[: -config.observation_trim]
    if not len(observed):
        raise ValueError("Observation stride/trim leaves no data.")
    mdb = MdbExomol(str(args.mdb_path), nurange=nu)
    opa = OpaPremodit.from_snapshot(
        mdb.to_snapshot(),
        nu,
        auto_trange=(config.temperature_min, config.temperature_max),
        diffmode=config.premodit_diffmode,
        dit_grid_resolution=config.broadening_resolution,
    )
    context = _context(nu, observed, opa, args.cia_path, config)
    context["source_info"] = {
        "molecular_database": str(args.mdb_path.resolve()),
        "dataset": "CO/12C-16O/Li2015",
        "note": "The saved opacity archive and its metadata are the execution inputs.",
    }
    return context


def _save_opacity(opacity, path):
    from exojax.opacity import saveopa

    saveopa(opacity, str(path), format="npz")


def _definition(metadata):
    return {
        name: metadata[name]
        for name in (
            "config",
            "priors",
            "truth",
            "units",
            "opacity",
            "inputs",
            "observation_seed",
            "probe_parameters",
        )
    }


def _write_case(directory, state, context, case_config, seed):
    storage = _storage()
    _save_opacity(context["opa"], directory / "premodit.npz")
    truth = mock_truth(case_config.noise_sigma)
    forward = make_forward(context)
    mean = np.asarray(predict(forward, truth))
    if mean.shape != np.shape(context["nu_obs"]) or not np.all(np.isfinite(mean)):
        raise ValueError("Mock spectrum must be finite and match the observation grid.")
    observed = mean + np.random.default_rng(seed).normal(
        0, case_config.noise_sigma, mean.shape
    )
    probes = [
        truth,
        *[
            {
                name: float(value)
                for name, value in unit_to_physical(
                    np.full(len(PRIOR_SPECS), fraction)
                ).items()
            }
            for fraction in (0.25, 0.75)
        ],
    ]
    flux = np.asarray([predict(forward, point) for point in probes])
    log_likelihood = np.asarray(
        [physical_log_likelihood(forward, observed, point) for point in probes]
    )
    log_prior = np.asarray([normalized_log_prior(point) for point in probes])
    if not all(
        np.all(np.isfinite(value)) for value in (flux, log_likelihood, log_prior)
    ):
        raise ValueError("Fixed-point predictions and densities must be finite.")
    storage.write_npz(
        directory / "case.npz",
        nu_grid=np.asarray(context["nu_grid"]),
        nu_obs=np.asarray(context["nu_obs"]),
        mean_flux=mean,
        observed_flux=observed,
        noise_sigma=np.asarray(case_config.noise_sigma),
        probe_flux=flux,
        probe_log_likelihood=log_likelihood,
        probe_log_prior=log_prior,
    )
    cia_path = Path(context["cia_path"]).resolve()
    state.update(
        case_kind="co_sampler_comparison",
        config=asdict(case_config),
        priors=PRIOR_SPECS,
        truth=truth,
        units=UNITS,
        observation_seed=seed,
        probe_parameters=probes,
        opacity={
            name: storage.sha256(directory / name)
            for name in ("premodit.npz", "premodit_metadata.json")
        },
        inputs={
            "cia_path": str(cia_path),
            "cia_sha256": storage.sha256(cia_path),
            "source_info": context.get("source_info"),
        },
        artifacts={"case_sha256": storage.sha256(directory / "case.npz")},
        observation_operator="Original CO tutorial: 8-stream ibased emission, rotation, Gaussian instrument profile, RV and point sampling; inferred exponential noise width.",
    )
    state["model_sha256"] = _digest(_definition(state))


def prepare(args):
    import jax

    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {}
    with _record_execution(directory / "prepare.json", state):
        jax.config.update("jax_enable_x64", True)
        state["provenance"] = _provenance(args, [args.cia_path])
        state["environment"] = _environment()
        config = CaseConfig(
            number_of_wavenumbers=args.number_of_wavenumbers,
            number_of_layers=args.number_of_layers,
            observation_stride=args.observation_stride,
            observation_trim=args.observation_trim,
        )
        _stage(directory / "prepare.json", state, "opacity_and_observation")
        context = _prepare_context(args, config)
        _write_case(directory, state, context, config, args.seed)
    print(f"Prepared shared CO observation: {directory}")


def load_case(directory):
    directory = Path(directory).resolve()
    storage = _storage()
    metadata = storage.read_metadata(directory / "prepare.json")
    if metadata.get("case_kind") != "co_sampler_comparison" or metadata.get(
        "model_sha256"
    ) != _digest(_definition(metadata)):
        raise ValueError("Case model/configuration digest does not match prepare.json.")
    if metadata["priors"] != PRIOR_SPECS or metadata["units"] != UNITS:
        raise ValueError("Saved prior or units differ from the shared tutorial model.")
    if set(metadata["opacity"]) != {"premodit.npz", "premodit_metadata.json"}:
        raise ValueError("Saved opacity manifest differs from the prepared format.")
    for filename, digest in metadata["opacity"].items():
        if storage.sha256(directory / storage.validate_run_id(filename)) != digest:
            raise ValueError("Saved opacity digest does not match prepare.json.")
    if (
        storage.sha256(Path(metadata["inputs"]["cia_path"]))
        != metadata["inputs"]["cia_sha256"]
    ):
        raise ValueError("CIA digest does not match prepare.json.")
    if storage.sha256(directory / "case.npz") != metadata["artifacts"]["case_sha256"]:
        raise ValueError("Observation/case digest does not match prepare.json.")
    with np.load(directory / "case.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    config = CaseConfig(**metadata["config"])
    grid = arrays["nu_grid"]
    observed_grid = grid[:: config.observation_stride]
    if config.observation_trim:
        observed_grid = observed_grid[: -config.observation_trim]
    size = len(observed_grid)
    shapes = {
        "nu_grid": (config.number_of_wavenumbers,),
        "nu_obs": (size,),
        "mean_flux": (size,),
        "observed_flux": (size,),
        "noise_sigma": (),
        "probe_flux": (3, size),
        "probe_log_likelihood": (3,),
        "probe_log_prior": (3,),
    }
    if (
        size == 0
        or set(arrays) != set(shapes)
        or any(arrays[name].shape != shape for name, shape in shapes.items())
        or not all(np.all(np.isfinite(array)) for array in arrays.values())
        or not np.all(np.diff(grid) > 0)
        or not np.all(grid > 0)
        or not np.array_equal(arrays["nu_obs"], observed_grid)
        or float(arrays["noise_sigma"]) != config.noise_sigma
        or metadata["truth"]["sigmain"] != config.noise_sigma
        or len(metadata["probe_parameters"]) != 3
    ):
        raise ValueError("Saved observation arrays are nonfinite or inconsistent.")
    return metadata, arrays


def build_context(directory, metadata, arrays):
    from exojax.opacity import OpaPremodit

    opacity = OpaPremodit.from_saved_opa(str(Path(directory) / "premodit.npz"))
    if not np.array_equal(np.asarray(opacity.nu_grid), arrays["nu_grid"]):
        raise ValueError("Saved opacity and observation model grids differ.")
    return _context(
        arrays["nu_grid"],
        arrays["nu_obs"],
        opacity,
        metadata["inputs"]["cia_path"],
        CaseConfig(**metadata["config"]),
    )


def load_context(directory):
    metadata, arrays = load_case(directory)
    return metadata, arrays, build_context(directory, metadata, arrays)


def _check_fixed_points(forward, metadata, arrays):
    probes = metadata["probe_parameters"]
    actual = {
        "probe_flux": np.asarray([predict(forward, point) for point in probes]),
        "probe_log_likelihood": np.asarray(
            [
                physical_log_likelihood(forward, arrays["observed_flux"], point)
                for point in probes
            ]
        ),
        "probe_log_prior": np.asarray(
            [normalized_log_prior(point, metadata["priors"]) for point in probes]
        ),
    }
    if not all(
        np.allclose(value, arrays[name], rtol=1e-9, atol=1e-8)
        for name, value in actual.items()
    ):
        raise ValueError(
            "Forward/likelihood/prior fixed points differ from the prepared model."
        )
    return {
        "passed": True,
        "rtol": 1e-9,
        "atol": 1e-8,
        "parameters": probes,
        **{name: value.tolist() for name, value in actual.items()},
    }


def _run_nuts(forward, observation, metadata, args):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    model = make_numpyro_model(forward, metadata["priors"])
    kernel = NUTS(model, forward_mode_differentiation=False)
    sampler = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    started = time.perf_counter()
    sampler.run(
        jax.random.PRNGKey(args.seed),
        spectrum=jnp.asarray(observation),
        extra_fields=("num_steps", "accept_prob"),
    )
    samples = sampler.get_samples(group_by_chain=True)
    extra = sampler.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((samples, extra))
    samples, extra = jax.device_get((samples, extra))
    elapsed = time.perf_counter() - started
    report = _helper("benchmark_inference").posterior_diagnostics(samples, extra)
    report.update(
        elapsed_seconds=elapsed,
        timing_scope="Compilation, warmup, sampling and host transfer; excludes construction and ArviZ diagnostics.",
    )
    arrays = {f"samples__{name}": value for name, value in samples.items()}
    arrays.update({f"extra__{name}": value for name, value in extra.items()})
    return report, arrays


def _run_jaxns(forward, observation, metadata, args):
    from _compare_samplers_jaxns import run_nested

    return run_nested(
        lambda point: physical_log_likelihood(forward, observation, point),
        metadata["priors"],
        seed=args.seed,
        num_live_points=args.num_live_points,
        max_samples=args.max_samples,
        dlogz=args.dlogz,
    )


def _run_directory(directory, method, run_id):
    if method not in ("nuts", "jaxns"):
        raise ValueError("Method must be nuts or jaxns.")
    return (
        Path(directory).resolve() / "runs" / _storage().validate_run_id(run_id) / method
    )


def run(args):
    import jax

    directory = args.output_dir.resolve()
    target = _run_directory(directory, args.method, args.run_id)
    target.mkdir(parents=True, exist_ok=False)
    state = {
        "method": args.method,
        "run_id": args.run_id,
        "seed": args.seed,
        "settings": {
            name: getattr(args, name)
            for name in (
                "num_chains",
                "num_warmup",
                "num_samples",
                "num_live_points",
                "max_samples",
                "dlogz",
            )
        },
        "nuts_quality_rules": dict(_helper("benchmark_inference").DEFAULT_RULES),
    }
    storage = _storage()
    with _record_execution(target / "result.json", state):
        jax.config.update("jax_enable_x64", True)
        state["provenance"] = _provenance(args, [directory / "prepare.json"])
        state["environment"] = _environment()
        _stage(target / "result.json", state, "input_validation")
        metadata, arrays, context = load_context(directory)
        state.update(
            model_sha256=metadata["model_sha256"],
            case_sha256=metadata["artifacts"]["case_sha256"],
            prepare_sha256=storage.sha256(directory / "prepare.json"),
        )
        forward = make_forward(context)
        state["fixed_points"] = _check_fixed_points(forward, metadata, arrays)
        _stage(target / "result.json", state, "sampling")
        start = time.perf_counter()
        report, samples = (_run_nuts if args.method == "nuts" else _run_jaxns)(
            forward, arrays["observed_flux"], metadata, args
        )
        state["sampler_setup_run_and_diagnostics_seconds"] = time.perf_counter() - start
        state["timing_scope"] = (
            "Sampler construction, compilation, sampling, host transfer and diagnostics; excludes case loading, fixed-point checks and artifact writing."
        )
        if args.method == "jaxns":
            report.setdefault("parameter_order", list(metadata["priors"]))
        state["nuts" if args.method == "nuts" else "nested"] = report
        _stage(target / "result.json", state, "sample_save")
        storage.write_npz(target / "samples.npz", **samples)
        state["samples"] = {
            "filename": "samples.npz",
            "sha256": storage.sha256(target / "samples.npz"),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": value.dtype.str}
                for name, value in samples.items()
            },
        }
    print(f"Saved {args.method} run: {target}")


def load_run(directory, method, run_id):
    metadata, _ = load_case(directory)
    target = _run_directory(directory, method, run_id)
    storage = _storage()
    report = storage.read_metadata(target / "result.json")
    if report.get("method") != method or report.get("run_id") != run_id:
        raise ValueError("Run ID/method differs from the explicit selection.")
    for name, expected in (
        ("model_sha256", metadata["model_sha256"]),
        ("case_sha256", metadata["artifacts"]["case_sha256"]),
        ("prepare_sha256", storage.sha256(Path(directory) / "prepare.json")),
    ):
        if report.get(name) != expected:
            raise ValueError(f"Run {name} differs from the prepared case.")
    if storage.sha256(target / "samples.npz") != report["samples"]["sha256"]:
        raise ValueError("Saved sample digest differs from the run.")
    with np.load(target / "samples.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    if set(arrays) != set(report["samples"]["arrays"]):
        raise ValueError("Saved sample fields differ from the manifest.")
    for name, array in arrays.items():
        specification = report["samples"]["arrays"][name]
        if (
            list(array.shape) != specification["shape"]
            or array.dtype.str != specification["dtype"]
        ):
            raise ValueError("Sample shape/dtype differs from the manifest.")
    if {
        name.removeprefix("samples__")
        for name in arrays
        if name.startswith("samples__")
    } != set(metadata["priors"]):
        raise ValueError("Saved posterior parameters differ from the shared prior.")
    if method == "nuts":
        expected_shape = (
            report["settings"]["num_chains"],
            report["settings"]["num_samples"],
        )
        if any(array.shape != expected_shape for array in arrays.values()):
            raise ValueError("Sample chain/draw shape differs from the run settings.")
        samples = {
            name.removeprefix("samples__"): value
            for name, value in arrays.items()
            if name.startswith("samples__")
        }
        extra = {
            name.removeprefix("extra__"): value
            for name, value in arrays.items()
            if name.startswith("extra__")
        }
        report["nuts"].update(
            _helper("benchmark_inference").posterior_diagnostics(
                samples, extra, rules=report["nuts_quality_rules"]
            )
        )
    else:
        from _compare_samplers_jaxns import validate_saved_results

        report["nested"] = validate_saved_results(
            report["nested"], arrays, metadata["priors"]
        )
    return report, arrays


def summarize(args):
    directory = args.output_dir.resolve()
    target = directory / "runs" / _storage().validate_run_id(args.run_id)
    target.mkdir(parents=True, exist_ok=True)
    state = {"run_id": args.run_id}
    with _record_execution(target / "comparison.json", state):
        nuts, chains = load_run(directory, "nuts", args.run_id)
        nested, weighted = load_run(directory, "jaxns", args.run_id)
        if nuts["provenance"]["code_sha256"] != nested["provenance"]["code_sha256"]:
            raise ValueError(
                "Sampler execution code differs; rerun the same shared model revision."
            )
        samples = {
            name.removeprefix("samples__"): value
            for name, value in chains.items()
            if name.startswith("samples__")
        }
        weights = weighted["weights"]
        means = {
            name: {
                "nuts": nuts["nuts"]["per_parameter"][name]["mean"],
                "jaxns": float(np.sum(weighted[f"samples__{name}"] * weights))
                if nested["nested"]["finite"]
                else None,
            }
            for name in samples
        }
        state.update(
            nuts=nuts,
            jaxns=nested,
            posterior_means=means,
            same_numeric_environment=all(
                nuts["environment"].get(name) == nested["environment"].get(name)
                for name in ("backend", "device_kind", "jax_enable_x64")
            )
            and all(
                nuts["environment"]["versions"][name]
                == nested["environment"]["versions"][name]
                for name in ("jax", "jaxlib", "numpy")
            ),
            sampler_setup_run_and_diagnostics_seconds={
                "nuts": nuts["sampler_setup_run_and_diagnostics_seconds"],
                "jaxns": nested["sampler_setup_run_and_diagnostics_seconds"],
            },
            quality={
                "nuts_passed": nuts["nuts"]["quality_passed"],
                "nested_converged": nested["nested"]["converged"],
            },
            diagnostic_scope="NUTS chain R-hat/bulk/tail ESS and nested normalized-weight ESS/evidence/stopping conditions are separate. Equal-weight resample count is not ESS; NUTS timing does not measure evidence capability.",
        )
    print(f"Saved sampler comparison: {target / 'comparison.json'}")


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative.")
    return value


def _positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive.")
    return value


def _sampler_seed(value):
    value = _nonnegative_int(value)
    if value >= 2**32:
        raise argparse.ArgumentTypeError("Sampler seed must fit in uint32.")
    return value


def _run_id(value):
    try:
        return _storage().validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser(
        "prepare", help="Create one immutable seeded CO observation and saved opacity."
    )
    preparation.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    preparation.add_argument(
        "--mdb-path", type=Path, default=Path(".database/CO/12C-16O/Li2015")
    )
    preparation.add_argument(
        "--cia-path", type=Path, default=Path(".database/H2-H2_2011.cia")
    )
    preparation.add_argument("--seed", type=_nonnegative_int, default=0)
    preparation.add_argument(
        "--number-of-wavenumbers", type=_positive_int, default=3500
    )
    preparation.add_argument("--number-of-layers", type=_positive_int, default=100)
    preparation.add_argument("--observation-stride", type=_positive_int, default=5)
    preparation.add_argument("--observation-trim", type=_nonnegative_int, default=50)
    preparation.set_defaults(handler=prepare)
    sampling = commands.add_parser(
        "run", help="Execute one sampler on the saved case in a fresh process."
    )
    sampling.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sampling.add_argument("--method", choices=("nuts", "jaxns"), required=True)
    sampling.add_argument("--run-id", type=_run_id, required=True)
    sampling.add_argument("--seed", type=_sampler_seed, default=0)
    sampling.add_argument("--num-chains", type=_positive_int, default=4)
    sampling.add_argument("--num-warmup", type=_positive_int, default=500)
    sampling.add_argument("--num-samples", type=_positive_int, default=1000)
    sampling.add_argument("--num-live-points", type=_positive_int, default=128)
    sampling.add_argument("--max-samples", type=_positive_int, default=10000)
    sampling.add_argument("--dlogz", type=_positive_float, default=0.01)
    sampling.set_defaults(handler=run)
    summary = commands.add_parser(
        "summarize",
        help="Reload raw chains/weights and report sampler-specific diagnostics.",
    )
    summary.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    summary.add_argument("--run-id", type=_run_id, required=True)
    summary.set_defaults(handler=summarize)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.handler(arguments)
