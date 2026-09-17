"""Fit two saved mock observations of one CO atmosphere: prepare/run/summarize.

The default case uses three synthetic lines and synthetic CIA, without a
database download. --co-case reuses a prepared PR4 CO case instead. These are
new mock observations; the source case's old observation is not fitted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

import compare_samplers as common
from _co_retrieval import CaseConfig, UNITS, normalized_log_prior, unit_to_physical
import _joint_observation as joint


DEFAULT_OUTPUT = common.ROOT / "output/joint_observation"
CASE_KIND = "joint_co_mock_v1"
CASE_UNITS = {name: value for name, value in UNITS.items() if name != "sigmain"}
CASE_UNITS.update(
    offset_b=UNITS["flux"], scale_a="dimensionless", scale_b="dimensionless"
)


def _provenance(args, inputs):
    return common._provenance(
        args,
        [Path(__file__), Path(joint.__file__), *inputs],
    )


def _synthetic_source(directory, args):
    """Build a small genuine PreMODIT/RT case with explicitly synthetic inputs."""
    from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
    from exojax.opacity import OpaPremodit
    from exojax.utils.grids import wavenumber_grid

    directory.mkdir()
    config = replace(
        CaseConfig(),
        number_of_wavenumbers=512,
        number_of_layers=8,
        observation_stride=8,
        observation_trim=0,
    )
    state = {}
    with common._record_execution(directory / "prepare.json", state):
        nu, _, _ = wavenumber_grid(
            config.wavelength_min,
            config.wavelength_max,
            config.number_of_wavenumbers,
            unit="AA",
            xsmode="premodit",
        )
        lines = np.interp([0.35, 0.5, 0.65], [0, 1], [nu[0], nu[-1]])
        snapshot = MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol",
                molmass=28.0,
                T_gQT=np.asarray([200.0, 500.0, 1000.0, 2000.0, 3000.0]),
                gQT=np.asarray([70.0, 180.0, 380.0, 850.0, 1400.0]),
            ),
            lines=Lines(
                nu_lines=lines,
                elower=np.asarray([200.0, 500.0, 900.0]),
                line_strength_ref_original=np.asarray([2e-22, 3e-22, 1e-22]),
            ),
            n_Texp=np.asarray([0.5, 0.55, 0.6]),
            alpha_ref=np.asarray([0.05, 0.06, 0.07]),
        )
        cia_path = directory / "H2-H2_SYNTHETIC.cia"
        cia_nu = np.linspace(nu[0] - 10, nu[-1] + 10, 16)
        with cia_path.open("w") as stream:
            for temperature in (200.0, 3000.0):
                stream.write(
                    f"H2-H2 {cia_nu[0]} {cia_nu[-1]} {len(cia_nu)} {temperature} 1e-46\n"
                )
                for value in cia_nu:
                    stream.write(f"{value} 1e-46\n")
        state["provenance"] = _provenance(args, [cia_path])
        state["environment"] = common._environment()
        opacity = OpaPremodit.from_snapshot(
            snapshot,
            nu,
            auto_trange=(config.temperature_min, config.temperature_max),
            diffmode=config.premodit_diffmode,
            dit_grid_resolution=config.broadening_resolution,
        )
        context = common._context(nu, nu[::8], opacity, cia_path, config)
        context["source_info"] = {
            "dataset": "Synthetic three-line CO and constant H2-H2 CIA",
            "line_wavenumbers": lines.tolist(),
            "scope": "Small wiring case, not a physical CO database approximation benchmark.",
        }
        common._write_case(directory, state, context, config, args.seed)
    return directory, *common.load_case(directory), context


def _source_context(args, directory):
    if args.co_case is None:
        return _synthetic_source(directory / "co_source", args)
    source = args.co_case.resolve()
    return source, *common.load_context(source)


def _definition(metadata):
    return {
        name: metadata[name]
        for name in (
            "case_kind",
            "source",
            "instruments",
            "priors",
            "units",
            "fixed_parameters",
            "truth",
            "observation_seed",
            "probe_parameters",
            "observation_operator",
        )
    }


def _local_identifiability(forward, arrays, truth):
    """Check local nuisance directions with the atmospheric parameters fixed."""
    import jax
    import jax.numpy as jnp

    names = ["offset_b", "scale_a", "scale_b"]

    def moments(values):
        parameters = {**truth, **dict(zip(names, values))}
        means = forward(parameters)
        errors = joint.error_scales(arrays, parameters)
        return jnp.concatenate([means["a"], means["b"], errors["a"], errors["b"]])

    jacobian = np.asarray(jax.jacfwd(moments)(jnp.asarray([truth[n] for n in names])))
    lengths = np.linalg.norm(jacobian, axis=0)
    if not np.isfinite(jacobian).all() or np.any(lengths == 0):
        raise ValueError("Instrument nuisance directions are nonfinite or inactive.")
    singular_values = np.linalg.svd(jacobian / lengths, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return {
        "parameter_order": names,
        "rank": rank,
        "expected_rank": len(names),
        "passed": rank == len(names),
        "singular_values": singular_values.tolist(),
        "tolerance": float(tolerance),
        "scope": "Local mean/noise Jacobian with atmosphere fixed and columns normalized; not global atmospheric identifiability or posterior convergence.",
    }


def _probe_values(forward, arrays, probes, priors):
    means = [forward(point) for point in probes]
    return {
        **{
            f"probe_mean_{name}": np.asarray([mean[name] for mean in means])
            for name in ("a", "b")
        },
        "probe_log_likelihood": np.asarray(
            [joint.log_likelihood(forward, arrays, point) for point in probes]
        ),
        "probe_log_prior": np.asarray(
            [normalized_log_prior(point, priors) for point in probes]
        ),
    }


def prepare(args):
    import jax

    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {}
    storage = common._storage()
    path = directory / "prepare.json"
    with common._record_execution(path, state):
        jax.config.update("jax_enable_x64", True)
        state["provenance"] = _provenance(args, [])
        state["environment"] = common._environment()
        common._stage(path, state, "source_preparation")
        source, source_metadata, _, context = _source_context(args, directory)
        common._stage(path, state, "joint_observation")
        arrays = joint.make_geometry(context)
        forward = joint.make_forward(context, arrays)
        truth = joint.mock_truth()
        means, errors = forward(truth), joint.error_scales(arrays, truth)
        generator = np.random.default_rng(args.seed)
        for name in ("a", "b"):
            arrays[f"mean_{name}"] = np.asarray(means[name])
            arrays[f"observed_{name}"] = means[name] + generator.normal(
                0.0, np.asarray(errors[name]), np.shape(means[name])
            )
        probes = [
            truth,
            *[
                {
                    name: float(value)
                    for name, value in unit_to_physical(
                        np.full(len(joint.JOINT_PRIORS), fraction), joint.JOINT_PRIORS
                    ).items()
                }
                for fraction in (0.25, 0.75)
            ],
        ]
        arrays.update(_probe_values(forward, arrays, probes, joint.JOINT_PRIORS))
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(
                "Prepared joint observations or fixed points are nonfinite."
            )
        state.update(
            case_kind=CASE_KIND,
            source={
                "directory": str(source.resolve()),
                "prepare_sha256": storage.sha256(source / "prepare.json"),
                "model_sha256": source_metadata["model_sha256"],
                "case_sha256": source_metadata["artifacts"]["case_sha256"],
                "observation_role": "Source observation is not used in the joint likelihood.",
            },
            instruments=joint.INSTRUMENTS,
            priors=joint.JOINT_PRIORS,
            units=CASE_UNITS,
            fixed_parameters={"offset_a": 0.0},
            truth=truth,
            observation_seed=args.seed,
            probe_parameters=probes,
            observation_operator="Shared CO RT and rotation, separate fixed Gaussian LSFs, shared RV, then top-hat wavenumber means. Instrument B combines adjacent instrument-A-sized bins after its own LSF; independent Gaussian errors, multiplicative error scales, additive B flux offset.",
            local_identifiability=_local_identifiability(forward, arrays, truth),
        )
        state["model_sha256"] = common._digest(_definition(state))
        storage.write_npz(directory / "case.npz", **arrays)
        state["case_sha256"] = storage.sha256(directory / "case.npz")
    print(f"Prepared two saved observations: {directory}")


def load_case(directory):
    directory = Path(directory).resolve()
    storage = common._storage()
    metadata = storage.read_metadata(directory / "prepare.json")
    if metadata.get("case_kind") != CASE_KIND or metadata.get(
        "model_sha256"
    ) != common._digest(_definition(metadata)):
        raise ValueError("Joint model/configuration digest differs from prepare.json.")
    if (
        metadata["priors"] != joint.JOINT_PRIORS
        or metadata["instruments"] != joint.INSTRUMENTS
        or metadata["fixed_parameters"] != {"offset_a": 0.0}
        or metadata["units"] != CASE_UNITS
    ):
        raise ValueError(
            "Joint priors, units, instruments or offset anchor differ from this case version."
        )
    source = Path(metadata["source"]["directory"])
    if storage.sha256(source / "prepare.json") != metadata["source"]["prepare_sha256"]:
        raise ValueError("Source prepare digest differs from the joint case.")
    source_metadata, _ = common.load_case(source)
    if (
        source_metadata["model_sha256"] != metadata["source"]["model_sha256"]
        or source_metadata["artifacts"]["case_sha256"]
        != metadata["source"]["case_sha256"]
    ):
        raise ValueError("Source model/observation digest differs from the joint case.")
    if storage.sha256(directory / "case.npz") != metadata["case_sha256"]:
        raise ValueError("Joint observation digest differs from prepare.json.")
    with np.load(directory / "case.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Joint observation arrays must be finite.")
    for name in ("a", "b"):
        size = joint.INSTRUMENTS[name]["num_bins"]
        for field in ("mean", "observed", "error"):
            if arrays[f"{field}_{name}"].shape != (size,):
                raise ValueError("Joint observation shape differs from the instrument.")
        if (
            arrays[f"bins_{name}"].shape != (size, 2)
            or arrays[f"probe_mean_{name}"].shape != (3, size)
            or np.any(arrays[f"error_{name}"] <= 0)
        ):
            raise ValueError("Joint bin/error/probe shapes are inconsistent.")
    if any(
        arrays[name].shape != (3,)
        for name in ("probe_log_likelihood", "probe_log_prior")
    ):
        raise ValueError("Joint fixed-point density shapes are inconsistent.")
    return metadata, arrays


def load_context(directory):
    metadata, arrays = load_case(directory)
    _, _, context = common.load_context(metadata["source"]["directory"])
    expected = joint.make_geometry(context, metadata["instruments"])
    if any(not np.array_equal(arrays[name], value) for name, value in expected.items()):
        raise ValueError(
            "Saved joint geometry/errors differ from the prepared instruments."
        )
    return metadata, arrays, context


def _check_fixed_points(forward, metadata, arrays):
    actual = _probe_values(
        forward, arrays, metadata["probe_parameters"], metadata["priors"]
    )
    if any(
        not np.allclose(value, arrays[name], rtol=1e-9, atol=1e-8)
        for name, value in actual.items()
    ):
        raise ValueError(
            "Joint forward/likelihood/prior fixed points differ from preparation."
        )
    return {"passed": True, "rtol": 1e-9, "atol": 1e-8}


def _run_nuts(forward, arrays, metadata, args):
    model = joint.make_numpyro_model(forward, arrays, metadata["priors"])
    observation = np.concatenate([arrays["observed_a"], arrays["observed_b"]])
    return common._run_numpyro_model(model, observation, args)


def _run_jaxns(forward, arrays, metadata, args):
    from _compare_samplers_jaxns import run_nested

    return run_nested(
        lambda parameters: joint.log_likelihood(forward, arrays, parameters),
        metadata["priors"],
        seed=args.seed,
        num_live_points=args.num_live_points,
        max_samples=args.max_samples,
        dlogz=args.dlogz,
    )


def run(args):
    import jax

    directory = args.output_dir.resolve()
    target = common._run_directory(directory, args.method, args.run_id)
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
        "nuts_quality_rules": dict(common._helper("benchmark_inference").DEFAULT_RULES),
    }
    storage, path = common._storage(), target / "result.json"
    with common._record_execution(path, state):
        jax.config.update("jax_enable_x64", True)
        state["provenance"] = _provenance(args, [directory / "prepare.json"])
        state["environment"] = common._environment()
        common._stage(path, state, "input_validation")
        metadata, arrays, context = load_context(directory)
        state.update(
            model_sha256=metadata["model_sha256"],
            case_sha256=metadata["case_sha256"],
            prepare_sha256=storage.sha256(directory / "prepare.json"),
        )
        forward = joint.make_forward(context, arrays, metadata["instruments"])
        state["fixed_points"] = _check_fixed_points(forward, metadata, arrays)
        common._stage(path, state, "sampling")
        started = time.perf_counter()
        report, raw = (_run_nuts if args.method == "nuts" else _run_jaxns)(
            forward, arrays, metadata, args
        )
        state["sampler_setup_run_and_diagnostics_seconds"] = (
            time.perf_counter() - started
        )
        state["timing_scope"] = (
            "Sampler construction, compilation, warmup/sampling and diagnostics; excludes input validation and artifact writing."
        )
        state["nuts" if args.method == "nuts" else "nested"] = report
        common._stage(path, state, "sample_save")
        state["samples"] = common._save_raw_samples(target, raw)
    print(f"Saved joint {args.method} run: {target}")


def load_run(directory, method, run_id):
    metadata, _ = load_case(directory)
    target = common._run_directory(directory, method, run_id)
    storage = common._storage()
    report = storage.read_metadata(target / "result.json")
    if report.get("method") != method or report.get("run_id") != run_id:
        raise ValueError("Joint run ID/method differs from the explicit selection.")
    for name, expected in (
        ("model_sha256", metadata["model_sha256"]),
        ("case_sha256", metadata["case_sha256"]),
        ("prepare_sha256", storage.sha256(Path(directory) / "prepare.json")),
    ):
        if report.get(name) != expected:
            raise ValueError(f"Joint run {name} differs from the prepared case.")
    return common._load_raw_samples(target, report, metadata["priors"])


def _posterior_summary(metadata, report, raw):
    nested = report["method"] == "jaxns"
    quality = (
        report["nested"]["converged"] if nested else report["nuts"]["quality_passed"]
    )
    parameters = {}
    for name in metadata["priors"]:
        if nested:
            values, weights = raw[f"samples__{name}"], raw["weights"]
            if report["nested"]["finite"]:
                order = np.argsort(values)
                indices = np.minimum(
                    np.searchsorted(np.cumsum(weights[order]), [0.05, 0.5, 0.95]),
                    len(values) - 1,
                )
                quantiles = values[order[indices]].tolist()
                mean = float(np.sum(values * weights))
            else:
                mean, quantiles = None, [None] * 3
        else:
            diagnostic = report["nuts"]["per_parameter"][name]
            mean = diagnostic["mean"]
            quantiles = [diagnostic["quantiles"][str(q)] for q in (0.05, 0.5, 0.95)]
        truth = metadata["truth"][name]
        parameters[name] = {
            "role": "instrument"
            if name in ("offset_b", "scale_a", "scale_b")
            else "shared_atmosphere",
            "truth": truth,
            "mean": mean,
            "q05_q50_q95": quantiles,
            "truth_in_90pct_interval": None
            if any(q is None for q in quantiles)
            else bool(quantiles[0] <= truth <= quantiles[2]),
        }
    return {
        "method": report["method"],
        "run_id": report["run_id"],
        "seed": report["seed"],
        "quality_passed": quality,
        "parameters": parameters,
        "diagnostics": report["nested" if nested else "nuts"],
        "local_identifiability": metadata["local_identifiability"],
        "fixed_parameters": metadata["fixed_parameters"],
        "instruments": metadata["instruments"],
        "quantile_definition": "Inverse normalized-weight empirical CDF."
        if nested
        else "NumPy linear pooled-chain quantiles.",
        "recovery_scope": "Descriptive intervals for one mock realization, conditional on sampler quality; not repeated-mock coverage or global identifiability. Short smoke runs do not establish recovery.",
    }


def summarize(args):
    directory = args.output_dir.resolve()
    target = common._run_directory(directory, args.method, args.run_id)
    target.mkdir(parents=True, exist_ok=True)
    storage, state = common._storage(), {}
    with common._record_execution(target / "summary.json", state):
        metadata, _ = load_case(directory)
        report, raw = load_run(directory, args.method, args.run_id)
        state.update(_posterior_summary(metadata, report, raw))
        state["inputs"] = {
            "result_sha256": storage.sha256(target / "result.json"),
            "samples_sha256": report["samples"]["sha256"],
            "prepare_sha256": report["prepare_sha256"],
        }
    print(f"Saved joint inference summary: {target / 'summary.json'}")


def load_summary(directory, method, run_id):
    """Load the selected summary only when its source run still matches."""
    target = common._run_directory(directory, method, run_id)
    storage = common._storage()
    saved = storage.read_metadata(target / "summary.json")
    report, raw = load_run(directory, method, run_id)
    expected = {
        "result_sha256": storage.sha256(target / "result.json"),
        "samples_sha256": report["samples"]["sha256"],
        "prepare_sha256": report["prepare_sha256"],
    }
    if (
        saved.get("inputs") != expected
        or saved.get("method") != method
        or saved.get("run_id") != run_id
    ):
        raise ValueError("Saved summary inputs differ from the selected joint run.")
    metadata, _ = load_case(directory)
    saved.update(_posterior_summary(metadata, report, raw))
    return saved


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser(
        "prepare", help="Save two new seeded mock observations."
    )
    preparation.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    preparation.add_argument(
        "--co-case",
        type=Path,
        help="Existing PR4 prepared CO case; otherwise use a small synthetic source.",
    )
    preparation.add_argument("--seed", type=common._nonnegative_int, default=0)
    preparation.set_defaults(handler=prepare)
    sampling = commands.add_parser(
        "run", help="Fit the joint case in a fresh sampler process."
    )
    sampling.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sampling.add_argument("--method", choices=("nuts", "jaxns"), default="nuts")
    sampling.add_argument("--run-id", type=common._run_id, required=True)
    sampling.add_argument("--seed", type=common._sampler_seed, default=0)
    sampling.add_argument("--num-chains", type=common._positive_int, default=4)
    sampling.add_argument("--num-warmup", type=common._positive_int, default=500)
    sampling.add_argument("--num-samples", type=common._positive_int, default=1000)
    sampling.add_argument("--num-live-points", type=common._positive_int, default=128)
    sampling.add_argument("--max-samples", type=common._positive_int, default=10000)
    sampling.add_argument("--dlogz", type=common._positive_float, default=0.01)
    sampling.set_defaults(handler=run)
    summary = commands.add_parser(
        "summarize",
        help="Reload raw samples and report quality and mock-truth intervals.",
    )
    summary.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    summary.add_argument("--method", choices=("nuts", "jaxns"), default="nuts")
    summary.add_argument("--run-id", type=common._run_id, required=True)
    summary.set_defaults(handler=summarize)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.handler(arguments)
