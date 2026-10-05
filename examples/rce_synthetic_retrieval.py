"""Reproducible, small TCE/RCE recovery with synthetic H2O/CO opacity.

Run ``python examples/rce_synthetic_retrieval.py --output output/rce_p6.json``.
Requires the optional ExoGibbs standard thermodynamics provider. This example
uses a local linear uncertainty estimate, not posterior sampling or real data.
"""

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from _rce_synthetic import SyntheticColumn


SEED = 20261005
SIGMA = 2e-6  # Dimensionless planet/star flux ratio; 2 ppm in every bin.
FIT_TOLERANCE = 1e-8
MAX_EVALUATIONS = 100


def _settings(mode):
    if mode == "rce":
        names = ["log_metal_scale", "C/O", "log_irradiation_amplitude"]
        truth = np.array([0.0, 0.6, 0.0])
        half_width = np.array([0.15, 0.15, 0.15])
        offsets = np.array([[-0.6, 0.5, -0.4], [0.5, -0.6, 0.4], [0.4, 0.5, -0.5]])
    elif mode == "tce":
        names = [
            "log_metal_scale",
            "C/O",
            "log_Ttop_over_2500",
            "log_Tbottom_over_2500",
        ]
        truth = np.array([0.0, 0.6, np.log(2000.0 / 2500.0), np.log(3000.0 / 2500.0)])
        half_width = np.array([0.15, 0.15, 0.1, 0.1])
        offsets = np.array(
            [[-0.6, 0.5, -0.4, 0.4], [0.5, -0.6, 0.4, -0.4], [0.4, 0.5, -0.5, 0.5]]
        )
    else:
        raise ValueError("mode must be 'rce' or 'tce'")
    return (
        names,
        truth,
        truth - half_width,
        truth + half_width,
        truth + offsets * half_width,
    )


def _uncertainty(jacobian, fitted, truth, lower, upper):
    """Known-noise covariance from first derivatives; never regularize a rank loss."""
    width = upper - lower
    _, singular_values, vt = np.linalg.svd(jacobian * width, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-10))
    result = {
        "method": "local linear likelihood, known noise; no higher-order AD",
        "rank": rank,
        "scaled_singular_values": singular_values.tolist(),
        "rank_relative_tolerance": 1e-10,
        "covariance": None,
        "prior_dependent": True,
        "truth_within_3sigma": None,
    }
    if rank != fitted.size:
        result["reason"] = "rank deficient; box prior controls unconstrained directions"
        return result
    covariance = (vt.T / singular_values**2) @ vt
    covariance *= width[:, None] * width[None, :]
    std = np.sqrt(np.diag(covariance))
    # A local Gaussian extending outside the box cannot be treated as a
    # likelihood-dominated posterior uncertainty.
    prior_dependent = 3 * std >= np.minimum(fitted - lower, upper - fitted)
    result.update(
        covariance=covariance.tolist(),
        standard_deviation=std.tolist(),
        prior_dependent=bool(np.any(prior_dependent)),
        prior_dependent_parameters=prior_dependent.tolist(),
        truth_offset_over_std=((fitted - truth) / std).tolist(),
        truth_within_3sigma=bool(np.all(np.abs(fitted - truth) <= 3 * std)),
        reason="3 sigma reaches the box boundary"
        if np.any(prior_dependent)
        else "local likelihood uncertainty fits inside the box",
    )
    return result


def _state_report(column, mode, parameters):
    if mode == "rce":
        solved = column.solve(jnp.asarray(parameters))
        nodes = jnp.append(solved.state.temperature, solved.state.bottom_temperature)
        rce = {
            name: np.asarray(value).tolist()
            for name, value in solved.state._asdict().items()
        }
        rce.update(
            derivative_valid=bool(solved.derivative_valid),
            derivative_status=int(solved.derivative_status),
            complementarity_margin=np.asarray(solved.complementarity_margin).tolist(),
            linear_residual=float(solved.linear_residual),
        )
    else:
        nodes = column.tce_nodes(jnp.asarray(parameters))
        rce = None
    entropy = column.entropy(nodes, parameters[0], parameters[1])
    chemistry = entropy.chemistry
    return _jsonable(
        {
            "temperature_nodes_K": nodes,
            "elemental_abundances": chemistry.b,
            "metal_mass_fraction": chemistry.metal_mass_fraction,
            "chemistry_diagnostics": chemistry.diagnostics._asdict(),
            "specific_entropy_J_kg_K": entropy.specific_entropy,
            "entropy_status": entropy.status,
            "rce": rce,
        }
    )


def run_recovery(column, mode):
    """Fit three independent starts; retain every failed start and its reason."""
    names, truth, lower, upper, initial = _settings(mode)
    spectrum = column.rce_spectrum if mode == "rce" else column.tce_spectrum
    predict = jax.jit(spectrum)
    jacobian = jax.jit(jax.jacfwd(spectrum))
    timing = {}
    for name, function in (("spectrum", predict), ("spectrum_jacobian", jacobian)):
        timing[name] = {}
        for call in ("first_call_including_compilation", "cached_call"):
            before = perf_counter()
            value = jax.block_until_ready(function(jnp.asarray(truth)))
            timing[name][call] = perf_counter() - before
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{mode}: injected truth has invalid {name}")
        if name == "spectrum":
            truth_spectrum = np.asarray(value)
    sigma = np.full(truth_spectrum.shape, SIGMA)
    # Each model gets the same standardized noise realization.
    observed = (
        truth_spectrum + np.random.default_rng(SEED).normal(size=sigma.size) * sigma
    )
    runs = []
    for start in initial:
        counters = {
            "forward_calls": 0,
            "jacobian_calls": 0,
            "invalid_forward": 0,
            "invalid_jacobian": 0,
        }

        def residual(parameters):
            counters["forward_calls"] += 1
            predicted = np.asarray(predict(jnp.asarray(parameters)))
            if not np.all(np.isfinite(predicted)):
                counters["invalid_forward"] += 1
                run["invalid_parameters"] = parameters.tolist()
                raise FloatingPointError(
                    "invalid chemistry or RCE sensitivity certificate"
                )
            return (predicted - observed) / sigma

        def derivative(parameters):
            counters["jacobian_calls"] += 1
            value = np.asarray(jacobian(jnp.asarray(parameters))) / sigma[:, None]
            if not np.all(np.isfinite(value)):
                counters["invalid_jacobian"] += 1
                run["invalid_parameters"] = parameters.tolist()
                raise FloatingPointError("nonfinite spectral Jacobian")
            return value

        run = {"initial": start.tolist(), "success": False}
        try:
            fit = least_squares(
                residual,
                start,
                jac=derivative,
                bounds=(lower, upper),
                x_scale=upper - lower,
                max_nfev=MAX_EVALUATIONS,
                ftol=FIT_TOLERANCE,
                xtol=FIT_TOLERANCE,
                gtol=FIT_TOLERANCE,
            )
            predicted = np.asarray(predict(jnp.asarray(fit.x)))
            uncertainty = _uncertainty(fit.jac, fit.x, truth, lower, upper)
            spectral_error = (predicted - truth_spectrum) / sigma
            run.update(
                success=bool(fit.success),
                status=int(fit.status),
                message=fit.message,
                fitted=fit.x.tolist(),
                chi_square=float(fit.fun @ fit.fun),
                optimality=float(fit.optimality),
                uncertainty=uncertainty,
                fitted_spectrum=predicted.tolist(),
                spectrum_truth_error_norm_sigma=float(np.linalg.norm(spectral_error)),
                spectrum_truth_error_max_sigma=float(np.max(np.abs(spectral_error))),
            )
        except FloatingPointError as error:
            # Abort this start rather than returning a flat finite penalty that
            # least_squares might incorrectly declare a converged fit.
            run["message"] = str(error)
        run["evaluations"] = counters
        runs.append(run)
        print(f"{mode} start {len(runs)}: {run['message']}", flush=True)

    accepted = [run for run in runs if run["success"]]
    best = min(accepted, key=lambda run: run["chi_square"]) if accepted else None
    threshold = float(np.sqrt(chi2.ppf(0.9973, truth.size)))
    recovered = len(accepted) == len(runs) and all(
        not run["uncertainty"]["prior_dependent"]
        and run["uncertainty"]["truth_within_3sigma"]
        and run["spectrum_truth_error_norm_sigma"] <= threshold
        for run in accepted
    )
    return {
        "parameter_names": names,
        "truth": truth.tolist(),
        "timing_seconds": timing,
        "prior": {
            "distribution": "uniform box",
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "noise_model": "independent Normal",
        "seed": SEED,
        "sigma": sigma.tolist(),
        "truth_spectrum": truth_spectrum.tolist(),
        "observed_spectrum": observed.tolist(),
        "runs": runs,
        "failed_starts": len(runs) - len(accepted),
        "all_starts_recovered": bool(recovered),
        "recovery_criterion": {
            "parameter_sigma": 3,
            "spectrum_error_norm_sigma": threshold,
            "requires_likelihood_dominated_uncertainty": True,
        },
        "truth_state": _state_report(column, mode, truth),
        "best_fit_state": _state_report(column, mode, np.asarray(best["fitted"]))
        if best
        else None,
    }


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    return value


def _metadata(column):
    import exogibbs

    root = Path(__file__).resolve().parents[1]
    gibbs_root = Path(exogibbs.__file__).resolve().parents[2]

    def revision(path):
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [
        Path(__file__),
        *(
            root / "examples" / name
            for name in (
                "_rce_synthetic.py",
                "_rce_chemistry.py",
                "_rce_entropy.py",
                "_rce_observation.py",
            )
        ),
        *(
            root / "src/exojax/atm" / name
            for name in ("rce_device.py", "rce_device_implicit.py")
        ),
    ]
    chemistry = column.chemistry
    return {
        "scope": "synthetic coupling/recovery only; no real opacity tables or posterior sampling",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy", "exojax", "exogibbs")
        },
        "revisions": {"exojax": revision(root), "exogibbs": revision(gibbs_root)},
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()],
        "pressure_boundaries_bar": np.asarray(column.boundaries).tolist(),
        "wavenumber_cm_inverse": np.asarray(column.nu).tolist(),
        "integration_widths_cm_inverse": np.asarray(column.widths).tolist(),
        "observation_matrix": np.asarray(column.W).tolist(),
        "species": chemistry.species,
        "elements": chemistry.elements,
        "isotope_convention": chemistry.isotope_convention,
        "abundance_convention": "10**log_metal_scale multiplies metals/H; C/O keeps C+O; H/He fixed",
        "thermodynamics": _jsonable(column.entropy.thermodynamics.metadata),
        "temperature_range_K": chemistry.temperature_range,
        "pressure_range_bar": chemistry.pressure_range,
        "fixed_physics": {
            "area_ratio": column.area_ratio,
            "internal_flux_cgs": column.internal_flux,
            "star_temperature_K": 5800,
            "gravity_cgs": 1000,
            "incidence_mu": 0.5,
            "absorbers": [chemistry.species[i] for i in column.absorbers],
        },
        "tolerances": {
            "chemistry_epsilon": column.chemistry_epsilon,
            "conservation_rtol": column.conservation_rtol,
            "flux_atol": 0.01,
            "flux_rtol": 0.0,
            "stability_atol": 1e-9,
            "entropy_scale_J_kg_K": column.entropy.entropy_scale,
            "fit_ftol_xtol_gtol": FIT_TOLERANCE,
            "fit_max_nfev": MAX_EVALUATIONS,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/rce_p6_recovery.json")
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    column = SyntheticColumn()
    result = {
        "metadata": _metadata(column),
        "tce": run_recovery(column, "tce"),
        "rce": run_recovery(column, "rce"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {args.output}", flush=True)
    if not all(result[mode]["all_starts_recovered"] for mode in ("tce", "rce")):
        raise SystemExit(
            "Recovery criterion failed; inspect per-start results and prior dependence."
        )


if __name__ == "__main__":
    main()
