"""Optional, reproducible posterior diagnostics for the NUTS benchmark."""

from __future__ import annotations

import importlib
import warnings

import numpy as np


DEFAULT_RULES = {
    "min_chains": 4,
    "max_rhat": 1.01,
    "min_ess_bulk": 400.0,
    "min_ess_tail": 400.0,
    "max_divergences": 0,
}
QUANTILES = (0.05, 0.5, 0.95)
DEFINITIONS = {
    "rhat_rank": (
        "ArviZ rhat(method='rank'): maximum of rank-normalized split R-hat "
        "and folded rank-normalized split R-hat."
    ),
    "ess_bulk": "ArviZ ess(method='bulk'): rank-normalized split-chain bulk ESS.",
    "ess_tail": (
        "ArviZ ess(method='tail', prob=(0.05, 0.95)): minimum ESS of the "
        "5% and 95% quantile indicators."
    ),
    "mcse_mean": "ArviZ mcse(method='mean'): Markov-chain standard error of the mean.",
    "mcse_quantiles": (
        "ArviZ mcse(method='quantile', prob=q): Markov-chain quantile standard "
        "error at q=0.05, 0.5, 0.95."
    ),
    "quantiles": "NumPy quantile(method='linear') over pooled chains and draws.",
    "quality": (
        "All parameter components must have rank R-hat strictly below max_rhat, "
        "bulk and tail ESS at least their minima, and finite diagnostics; "
        "at least min_chains chains and no more than max_divergences divergences. "
        "Missing diagnostics, nonfinite samples, or constant chains fail quality."
    ),
    "scope": "Single-run diagnostics do not establish repeated-run coverage.",
}


def _finite(value):
    value = float(np.asarray(value).item())
    return value if np.isfinite(value) else None


def _rules(overrides):
    result = dict(DEFAULT_RULES)
    if overrides is not None:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError(f"Unknown posterior quality rules: {sorted(unknown)}")
        result.update(overrides)
    for name, value in result.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Invalid posterior quality rule {name}: {value!r}")
        if name in ("min_chains", "max_divergences"):
            minimum = 2 if name == "min_chains" else 0
            if int(value) != value or value < minimum:
                raise ValueError(f"Invalid posterior quality rule {name}: {value!r}")
            result[name] = int(value)
        elif value <= 0:
            raise ValueError(f"Invalid posterior quality rule {name}: {value!r}")
        else:
            result[name] = float(value)
    return result


def _components(samples):
    if not samples:
        raise ValueError("Posterior samples must contain at least one parameter.")
    shape = None
    components = {}
    for name, values in samples.items():
        values = np.asarray(values)
        if (
            not isinstance(name, str)
            or values.ndim < 2
            or values.dtype.kind not in "fiu"
        ):
            raise ValueError("Samples require named numeric (chain, draw, ...) arrays.")
        if shape is None:
            shape = values.shape[:2]
        if values.shape[:2] != shape or min(values.shape) < 1:
            raise ValueError("Sample chain/draw dimensions must match and be nonempty.")
        for index in np.ndindex(values.shape[2:]):
            label = name if not index else f"{name}[{','.join(map(str, index))}]"
            if label in components:
                raise ValueError(f"Duplicate posterior component label: {label}")
            components[label] = values[(slice(None), slice(None)) + index]
    return shape, components


def _divergences(extra, shape):
    values = extra.get("diverging")
    if values is None:
        return None, "Divergence flags are unavailable."
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind not in "bfiu":
        return None, "Divergence flags require the sample (chain, draw) shape."
    if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
        return None, "Divergence flags must contain only finite boolean values."
    return int(np.count_nonzero(values)), None


def posterior_diagnostics(samples, extra, rules=None):
    """Diagnose saved chains without changing JAX configuration or requiring ArviZ.

    ArviZ is an optional runtime dependency. Its absence leaves descriptive
    moments available but explicitly fails quality instead of substituting a
    different definition of effective sample size.
    """
    rules = _rules(rules)
    shape, components = _components(samples)
    divergence_count, divergence_error = _divergences(extra, shape)
    reasons = []
    if shape[0] < rules["min_chains"]:
        reasons.append(
            f"Requires at least {rules['min_chains']} chains; found {shape[0]}."
        )
    if shape[1] < 4:
        reasons.append("Rank diagnostics require at least four draws per chain.")
    if divergence_error:
        reasons.append(divergence_error)
    elif divergence_count > rules["max_divergences"]:
        reasons.append(f"Divergences exceed the limit: {divergence_count}.")
    implementation = {"name": "arviz", "version": None, "available": False}
    unavailable = None
    try:
        arviz = importlib.import_module("arviz")
        implementation.update(version=arviz.__version__, available=True)
    except ImportError as error:
        arviz = None
        unavailable = f"ArviZ diagnostics unavailable: {type(error).__name__}: {error}"
        reasons.append(unavailable)

    per_parameter = {}
    for name, values in components.items():
        finite = bool(np.all(np.isfinite(values)))
        record = {
            "mean": None,
            "quantiles": {str(q): None for q in QUANTILES},
            "mcse_mean": None,
            "mcse_quantiles": {str(q): None for q in QUANTILES},
            "rhat_rank": None,
            "ess_bulk": None,
            "ess_tail": None,
            "finite_samples": finite,
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
            "constant_chains": [],
            "quality_passed": False,
            "failure_reasons": [],
        }
        per_parameter[name] = record
        errors = record["failure_reasons"]
        if not finite:
            errors.append("Nonfinite posterior samples.")
            reasons.append(f"{name}: nonfinite posterior samples.")
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            record["mean"] = _finite(np.mean(values))
            record["quantiles"] = {
                str(q): _finite(value)
                for q, value in zip(QUANTILES, np.quantile(values, QUANTILES))
            }
        record["constant_chains"] = np.flatnonzero(
            np.all(values == values[:, :1], axis=1)
        ).tolist()
        if record["constant_chains"]:
            errors.append("At least one chain is constant.")
        if arviz is not None and min(shape) >= 2 and shape[1] >= 4:
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    record["rhat_rank"] = _finite(arviz.rhat(values, method="rank"))
                    record["ess_bulk"] = _finite(arviz.ess(values, method="bulk"))
                    record["ess_tail"] = _finite(
                        arviz.ess(values, method="tail", prob=(0.05, 0.95))
                    )
                    record["mcse_mean"] = _finite(arviz.mcse(values, method="mean"))
                    record["mcse_quantiles"] = {
                        str(q): _finite(arviz.mcse(values, method="quantile", prob=q))
                        for q in QUANTILES
                    }
            except (
                ValueError,
                TypeError,
                ZeroDivisionError,
                FloatingPointError,
            ) as error:
                errors.append(
                    f"Diagnostic evaluation failed: {type(error).__name__}: {error}"
                )
        diagnostics = [
            record["mean"],
            *record["quantiles"].values(),
            record["rhat_rank"],
            record["ess_bulk"],
            record["ess_tail"],
            record["mcse_mean"],
            *record["mcse_quantiles"].values(),
        ]
        if any(value is None for value in diagnostics):
            errors.append(
                "One or more moments or diagnostics are unavailable/nonfinite."
            )
        else:
            if record["rhat_rank"] >= rules["max_rhat"]:
                errors.append("Rank R-hat reaches or exceeds its strict upper limit.")
            if record["ess_bulk"] < rules["min_ess_bulk"]:
                errors.append("Bulk ESS is below its minimum.")
            if record["ess_tail"] < rules["min_ess_tail"]:
                errors.append("Tail ESS is below its minimum.")
        record["quality_passed"] = not errors
        reasons.extend(f"{name}: {error}" for error in errors)

    def aggregate(key, function):
        values = [record[key] for record in per_parameter.values()]
        return None if any(value is None for value in values) else function(values)

    return {
        "status": "unavailable" if unavailable else "completed",
        "quality_passed": not reasons,
        "failure_reasons": reasons,
        "num_chains": shape[0],
        "num_draws": shape[1],
        "rules": rules,
        "implementation": implementation,
        "definitions": dict(DEFINITIONS),
        "per_parameter": per_parameter,
        "summary": {
            "max_rhat_rank": aggregate("rhat_rank", max),
            "min_ess_bulk": aggregate("ess_bulk", min),
            "min_ess_tail": aggregate("ess_tail", min),
            "divergences": divergence_count,
        },
    }


def _difference(left, right, left_mcse, right_mcse):
    difference = None if left is None or right is None else _finite(right - left)
    combined = (
        None
        if left_mcse is None or right_mcse is None
        else _finite(np.hypot(left_mcse, right_mcse))
    )
    standardized = (
        _finite(difference / combined)
        if difference is not None and combined is not None and combined > 0
        else None
    )
    return {
        "difference": difference,
        "combined_mcse": combined,
        "standardized_difference": standardized,
    }


def posterior_difference(left_report, right_report):
    """Compare posterior summaries, without inventing an equivalence threshold."""
    left = left_report["per_parameter"]
    right = right_report["per_parameter"]
    if set(left) != set(right):
        raise ValueError(
            "Posterior comparisons require identical parameter components."
        )
    result = {}
    for name, first in left.items():
        second = right[name]
        result[name] = {
            "mean": _difference(
                first["mean"], second["mean"], first["mcse_mean"], second["mcse_mean"]
            ),
            "quantiles": {
                str(q): _difference(
                    first["quantiles"][str(q)],
                    second["quantiles"][str(q)],
                    first["mcse_quantiles"][str(q)],
                    second["mcse_quantiles"][str(q)],
                )
                for q in QUANTILES
            },
        }
    return {
        "direction": "right minus left",
        "mcse_definition": (
            "Combined MCSE is hypot(left MCSE, right MCSE), assuming independent "
            "Monte Carlo errors. Shared random seeds can violate this assumption. "
            "Standardized differences divide by this combined MCSE; no equivalence "
            "threshold or coverage claim is applied."
        ),
        "per_parameter": result,
    }


def predictive_summary(predictions, *, includes_noise=False):
    """Summarize supplied predictions with preserved observation coordinates."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 3 or min(predictions.shape) < 1:
        raise ValueError(
            "Predictions require a nonempty (chain, draw, observation) array."
        )
    finite = bool(np.all(np.isfinite(predictions)))
    mean = None
    quantiles = {str(q): None for q in QUANTILES}
    if finite:
        with np.errstate(over="ignore", invalid="ignore"):
            mean = [_finite(value) for value in np.mean(predictions, axis=(0, 1))]
            quantiles = {
                str(q): [_finite(value) for value in row]
                for q, row in zip(
                    QUANTILES, np.quantile(predictions, QUANTILES, axis=(0, 1))
                )
            }
        finite = all(
            value is not None for row in (mean, *quantiles.values()) for value in row
        )
    return {
        "finite": finite,
        "shape": list(predictions.shape),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(predictions))),
        "mean": mean,
        "quantiles": quantiles,
        "definition": (
            "Pointwise posterior distribution of replicated observations including measurement noise; pooled over chain/draw axes with linear quantiles."
            if includes_noise
            else "Pointwise posterior distribution of noiseless model predictions, "
            "pooled over the supplied chain/draw axes; measurement noise is not added. "
            "Quantiles use NumPy's linear interpolation."
        ),
    }
