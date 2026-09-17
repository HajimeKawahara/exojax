"""Optional JAXNS adapter for the shared CO retrieval example.

The public API and termination flags are checked against JAXNS 2.6.9.
Importing this module does not import JAXNS or change JAX configuration.
"""

from importlib import metadata
import time
from types import SimpleNamespace

import numpy as np


SUPPORTED_JAXNS_VERSION = "2.6.9"
TERMINATION_FLAGS = {
    1: "maximum_samples",
    2: "evidence_uncertainty",
    4: "remaining_evidence_dlogz",
    8: "effective_sample_size",
    16: "maximum_likelihood_evaluations",
    32: "likelihood_contour",
    64: "low_efficiency",
    128: "live_point_plateau",
    256: "relative_likelihood_spread",
    512: "absolute_likelihood_spread",
    1024: "no_seed_points",
    2048: "peak_evidence_fraction",
}


def _runtime():
    """Load only the explicitly tested optional API, with an actionable error."""
    try:
        version = metadata.version("jaxns")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "JAXNS is optional. Install jaxns==2.6.9 in a separate environment; "
            "the tested stack uses jax==0.6.2 and tfp-nightly."
        ) from exc
    if version != SUPPORTED_JAXNS_VERSION:
        raise RuntimeError(
            f"This adapter supports jaxns=={SUPPORTED_JAXNS_VERSION}; found {version}."
        )
    try:
        import jaxns
        from tensorflow_probability.substrates.jax import distributions
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "The optional JAXNS/JAX/TFP stack is incompatible. The tested stack is "
            "jaxns==2.6.9, jax==0.6.2, tfp-nightly==0.26.0.dev20260907."
        ) from exc
    return jaxns, distributions


def build_jaxns_model(log_likelihood, prior_specs):
    """Use normalized physical priors and their inverse CDFs on the unit cube.

    ``log_likelihood`` takes one physical-parameter dictionary. No prior density
    or unit-cube Jacobian is added to it: JAXNS integrates against the prior.
    """
    jaxns, distributions = _runtime()
    if not prior_specs:
        raise ValueError("At least one prior is required.")
    priors = {}
    for name, spec in prior_specs.items():
        if spec["distribution"] == "uniform":
            low, high = float(spec["low"]), float(spec["high"])
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError(f"Invalid Uniform prior for {name}.")
            # TFP defaults Python floats to float32 even with JAX x64 enabled.
            priors[name] = distributions.Uniform(
                low=np.float64(low), high=np.float64(high)
            )
        elif spec["distribution"] == "exponential":
            rate = float(spec["rate"])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"Invalid Exponential prior for {name}.")
            priors[name] = distributions.Exponential(rate=np.float64(rate))
        else:
            raise ValueError(f"Unsupported prior for {name}: {spec['distribution']}.")

    def prior_model():
        parameters = {}
        for name, distribution in priors.items():
            parameters[name] = yield jaxns.Prior(distribution, name=name)
        return (parameters,)

    return jaxns.Model(prior_model=prior_model, log_likelihood=log_likelihood)


def _summarize_results(results):
    """Preserve weighted draws and distinguish stopping from convergence."""
    arrays = {
        f"samples__{name}": np.asarray(value) for name, value in results.samples.items()
    }
    arrays.update(
        log_weights=np.asarray(results.log_dp_mean),
        log_likelihood=np.asarray(results.log_L_samples),
        unit_samples=np.asarray(results.U_samples),
        log_prior_volume=np.asarray(results.log_X_mean),
        num_live_points=np.asarray(results.num_live_points_per_sample),
        num_likelihood_evaluations=np.asarray(
            results.num_likelihood_evaluations_per_sample
        ),
    )
    log_weights = arrays["log_weights"]
    count = int(results.total_num_samples)
    valid_weights = (
        log_weights.shape == (count,)
        and count > 0
        and np.any(np.isfinite(log_weights))
        and not np.any(np.isnan(log_weights) | np.isposinf(log_weights))
    )
    weighted_ess = None
    if valid_weights:
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        weighted_ess = float(1.0 / np.sum(weights**2))
        arrays["weights"] = weights
    else:
        arrays["weights"] = np.full(log_weights.shape, np.nan)

    log_evidence = float(results.log_Z_mean)
    log_evidence_error = float(results.log_Z_uncert)
    jaxns_ess = float(results.ESS)
    finite = bool(
        valid_weights
        and np.isfinite([log_evidence, log_evidence_error, jaxns_ess]).all()
        and log_evidence_error >= 0
        and jaxns_ess > 0
        and all(
            value.shape[0] == count and np.isfinite(value).all()
            for name, value in arrays.items()
            if name.startswith("samples__") or name == "unit_samples"
        )
        and arrays["log_likelihood"].shape == (count,)
        and not np.any(
            np.isnan(arrays["log_likelihood"]) | np.isposinf(arrays["log_likelihood"])
        )
        and not np.any(np.isneginf(arrays["log_likelihood"]) & (arrays["weights"] > 0))
    )
    reason = int(results.termination_reason)
    labels = [label for flag, label in TERMINATION_FLAGS.items() if reason & flag]
    if reason & ~sum(TERMINATION_FLAGS):
        labels.append("unknown_termination_flag")
    # A resource limit, plateau or missing seeds is not evidence convergence.
    converged = bool(finite and reason == 4)
    report = {
        "status": "completed" if converged else "unconverged" if finite else "failed",
        "converged": converged,
        "finite": finite,
        "num_samples": count,
        "weighted_ess": weighted_ess if finite else None,
        "weighted_ess_definition": "1 / sum(normalized_saved_weights ** 2)",
        "jaxns_ess": jaxns_ess if finite else None,
        "log_evidence": log_evidence if finite else None,
        "log_evidence_error": log_evidence_error if finite else None,
        "termination_reason": reason,
        "termination_labels": labels,
        "total_likelihood_evaluations": int(results.total_num_likelihood_evaluations),
        "total_phantom_samples": int(results.total_phantom_samples),
        "weight_source": "JAXNS NestedSamplerResults.log_dp_mean, without resampling",
        "ess_scope": "Weight concentration; not an MCMC autocorrelation ESS.",
    }
    return report, arrays


def validate_saved_results(report, arrays, prior_specs):
    """Recompute quality from saved weights without importing either sampler.

    Invalid numerical outcomes stay failed. Inconsistent array structure,
    weights or finite prior transforms are corrupt records and are rejected.
    Evidence uncertainty itself cannot be re-estimated from posterior weights.
    """
    required = {
        "num_samples",
        "parameter_order",
        "termination_reason",
        "log_evidence",
        "log_evidence_error",
        "jaxns_ess",
    }
    if not required <= report.keys():
        raise ValueError("Saved nested report is missing required fields.")
    count, order = report["num_samples"], report["parameter_order"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or not isinstance(order, list)
        or not order
        or not all(isinstance(name, str) for name in order)
        or len(order) != len(prior_specs)
        or set(order) != set(prior_specs)
    ):
        raise ValueError("Saved nested sample count or parameter order is invalid.")
    if (
        not isinstance(report["termination_reason"], int)
        or report["termination_reason"] < 0
    ):
        raise ValueError("Saved nested termination reason is invalid.")
    for name in ("total_likelihood_evaluations", "total_phantom_samples"):
        if name in report and (not isinstance(report[name], int) or report[name] < 0):
            raise ValueError(f"Saved nested count is invalid: {name}.")
    sample_keys = {f"samples__{name}" for name in order}
    if {name for name in arrays if name.startswith("samples__")} != sample_keys:
        raise ValueError("Saved nested parameters differ from the prior.")
    required_arrays = sample_keys | {
        "log_weights",
        "weights",
        "log_likelihood",
        "unit_samples",
    }
    if not required_arrays <= arrays.keys():
        raise ValueError("Saved nested arrays are missing required fields.")
    for name, value in arrays.items():
        expected = (count, len(order)) if name == "unit_samples" else (count,)
        if value.shape != expected or value.dtype.kind not in "fiu":
            raise ValueError(f"Saved nested array shape or dtype is invalid: {name}.")
    unit = arrays["unit_samples"]
    if np.any(np.isfinite(unit) & ((unit < 0) | (unit > 1))):
        raise ValueError("Saved nested coordinates fall outside the unit cube.")
    finite_transform = True
    for index, name in enumerate(order):
        spec = prior_specs[name]
        with np.errstate(divide="ignore", invalid="ignore"):
            if spec["distribution"] == "uniform":
                expected = spec["low"] + unit[:, index] * (spec["high"] - spec["low"])
            elif spec["distribution"] == "exponential":
                expected = -np.log1p(-unit[:, index]) / spec["rate"]
            else:
                raise ValueError(f"Unsupported saved prior: {name}.")
        actual = arrays[f"samples__{name}"]
        finite_transform &= bool(np.isfinite(expected).all())
        finite = np.isfinite(expected) & np.isfinite(actual)
        if not np.allclose(actual[finite], expected[finite], rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Saved nested samples disagree with the prior transform/order."
            )

    def numeric(name):
        try:
            return np.nan if report[name] is None else float(report[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Saved nested diagnostic is not scalar: {name}."
            ) from error

    results = SimpleNamespace(
        samples={name: arrays[f"samples__{name}"] for name in order},
        log_dp_mean=arrays["log_weights"],
        log_L_samples=arrays["log_likelihood"],
        U_samples=unit,
        log_X_mean=arrays.get("log_prior_volume", np.zeros(count)),
        num_live_points_per_sample=arrays.get("num_live_points", np.zeros(count)),
        num_likelihood_evaluations_per_sample=arrays.get(
            "num_likelihood_evaluations", np.zeros(count)
        ),
        total_num_samples=count,
        log_Z_mean=numeric("log_evidence"),
        log_Z_uncert=numeric("log_evidence_error"),
        ESS=numeric("jaxns_ess") if finite_transform else np.nan,
        termination_reason=report["termination_reason"],
        total_num_likelihood_evaluations=report.get("total_likelihood_evaluations", 0),
        total_phantom_samples=report.get("total_phantom_samples", 0),
    )
    recomputed, expected_arrays = _summarize_results(results)
    if not np.allclose(
        arrays["weights"],
        expected_arrays["weights"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("Saved nested weights disagree with the original log weights.")
    if np.isfinite(expected_arrays["weights"]).all() and (
        np.any(arrays["weights"] < 0)
        or not np.isclose(arrays["weights"].sum(), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("Saved nested weights are not normalized probabilities.")
    return {**report, **recomputed}


def run_nested(
    log_likelihood,
    prior_specs,
    seed,
    num_live_points=128,
    max_samples=10000,
    dlogz=0.01,
):
    """Run one independent seed and retain original posterior weights.

    Timing includes compilation, execution and result conversion. The caller
    saves execution failures and supplies the same observation/model as NUTS.
    """
    if int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer.")
    if int(num_live_points) != num_live_points or num_live_points < 2:
        raise ValueError("num_live_points must be an integer of at least two.")
    if int(max_samples) != max_samples or max_samples <= num_live_points:
        raise ValueError("max_samples must be an integer larger than num_live_points.")
    if not np.isfinite(dlogz) or dlogz <= 0:
        raise ValueError("dlogz must be finite and positive.")
    import jax

    model = build_jaxns_model(log_likelihood, prior_specs)
    jaxns, _ = _runtime()
    sampler = jaxns.NestedSampler(
        model=model,
        num_live_points=int(num_live_points),
        max_samples=int(max_samples),
        # Evidence sampling uses no phantom draws and one explicit device.
        k=0,
        parameter_estimation=False,
        devices=[jax.devices()[0]],
        verbose=False,
    )
    condition = jaxns.TerminationCondition(
        dlogZ=float(dlogz), max_samples=int(max_samples)
    )
    started = time.perf_counter()
    reason, state = jax.jit(lambda key: sampler(key, term_cond=condition))(
        jax.random.PRNGKey(int(seed))
    )
    jax.block_until_ready(state)
    results = sampler.to_results(reason, state, trim=True)
    report, arrays = _summarize_results(results)
    report.update(
        seed=int(seed),
        parameter_order=list(prior_specs),
        prior_coordinate_system="Unit cube transformed by each normalized prior inverse CDF.",
        elapsed_seconds=time.perf_counter() - started,
        timing_scope="Compilation, sampling and conversion of one independent run.",
        termination_conditions={"dlogZ": float(dlogz), "max_samples": int(max_samples)},
        num_live_points=int(sampler.num_live_points),
        allocated_max_samples=int(sampler.nested_sampler.max_samples),
        versions={
            name: metadata.version(name)
            for name in ("jaxns", "jax", "jaxlib", "tfp-nightly", "numpy", "scipy")
        },
    )
    return report, arrays
