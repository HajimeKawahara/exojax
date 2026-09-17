"""Prepare and validate an offline, real CO/H2O CKD mixture case.

Run with JAX_ENABLE_X64=True. GPU execution is an explicit user choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from _ckd_mixture_case import (
    METHODS, PARAMETER_ORDER, ROOT, collect_provenance, load_context, make_forward,
    prepare_case, runtime_environment, sha256, write_json, write_npz,
)
from benchmark_metrics import derivative_difference, directional_check, observation_error
from diffgrid_nuts_storage import validate_run_id


def evaluate_context(context):
    """Keep scientific approximation error separate from local AD/FD agreement."""
    arrays, metadata = context["arrays"], context["metadata"]
    bounds, sigma = np.asarray(context["bounds"]), context["sigma"]
    widths = bounds[:, 1] - bounds[:, 0]
    budgets = metadata["budgets"]
    points = np.asarray(arrays["validation_points"])
    coordinates = (points - bounds[:, 0]) / widths
    refined = jax.jit(make_forward(context, "lbl", refined=True))
    forwards = {method: jax.jit(make_forward(context, method)) for method in METHODS}
    references = np.stack([np.asarray(refined(point)) for point in points])
    spectra = {method: np.stack([np.asarray(forward(point)) for point in points])
               for method, forward in forwards.items()}
    reference_checks = [
        observation_error(candidate, reference, sigma,
                          max_error=budgets["max_error_in_noise"] * budgets["reference_fraction"],
                          max_q=budgets["max_q"] * budgets["reference_fraction"] ** 2)
        for candidate, reference in zip(spectra["lbl"], references)
    ]
    report = {
        "reference": {"passed": all(row["passed"] for row in reference_checks),
                      "checks": reference_checks,
                      "scope": "Twice the LBL spectral midpoint resolution; same retained lines and atmospheric layers."},
        "methods": {}, "ng_sweep": {},
        "gradient_scope": "Prior-width-scaled coordinates; noise-whitened forward and same-data log likelihood. Local finite differences do not establish global smoothness.",
    }
    jacobians = {}
    for method, forward in forwards.items():
        print(f"Validating spectra and local gradients: {method}", flush=True)
        scaled = jax.jit(lambda u: forward(bounds[:, 0] + jnp.asarray(widths) * u) / sigma)
        likelihood = jax.jit(lambda u: -0.5 * jnp.sum((scaled(u) - context["observed"] / sigma) ** 2))
        gradient_rows = []
        for index, position in enumerate(coordinates):
            for name, direction in zip(PARAMETER_ORDER, np.eye(3)):
                row = {"point": index, "parameter": name}
                for label, function in (("forward", scaled), ("log_likelihood", likelihood)):
                    row[label] = directional_check(
                        function, position, direction, steps=budgets["steps"],
                        tolerance=budgets["gradient_tolerance"],
                        in_domain=lambda u: bool(np.all((u > 0) & (u < 1))),
                    )
                row["passed"] = row["forward"]["passed"] and row["log_likelihood"]["passed"]
                gradient_rows.append(row)
        jacobian = jax.jit(jax.jacfwd(scaled))
        jacobians[method] = np.stack([np.asarray(jacobian(position)) for position in coordinates])
        error_rows = [observation_error(candidate, reference, sigma,
                                        max_error=budgets["max_error_in_noise"], max_q=budgets["max_q"])
                      for candidate, reference in zip(spectra[method], references)]
        spectrum_passed = all(row["passed"] for row in error_rows)
        gradient_passed = all(row["passed"] for row in gradient_rows)
        report["methods"][method] = {
            "passed": report["reference"]["passed"] and spectrum_passed and gradient_passed,
            "spectrum_passed": spectrum_passed, "gradient_passed": gradient_passed,
            "spectrum_checks": error_rows, "gradient_checks": gradient_rows,
            "derivative_difference_from_lbl": [derivative_difference(candidate, reference)
                                               for candidate, reference in zip(jacobians[method], jacobians["lbl"])],
        }
    for ng in metadata["config"]["ng_values"]:
        report["ng_sweep"][str(ng)] = {}
        for method in ("premixed", "rorr", "same_g"):
            forward = jax.jit(make_forward(context, method, ng=ng))
            values = np.stack([np.asarray(forward(point)) for point in points])
            spectra[f"{method}_ng_{ng}"] = values
            report["ng_sweep"][str(ng)][method] = [
                observation_error(value, reference, sigma,
                                  max_error=budgets["max_error_in_noise"], max_q=budgets["max_q"])
                for value, reference in zip(values, references)
            ]
    saved = {"parameters": points, "reference_refined": references,
             "sigma": np.asarray(sigma), **{f"spectrum_{key}": value for key, value in spectra.items()},
             **{f"jacobian_{key}": value for key, value in jacobians.items()}}
    return report, saved


def validate_case(args):
    """Persist failures as well as passes; never overwrite a named validation."""
    if not jax.config.jax_enable_x64:
        raise ValueError("Enable JAX x64 before validation.")
    context = load_context(args.output_dir)
    validation_id = validate_run_id(args.validation_id)
    requested = list(args.methods)
    if not requested or any(method not in METHODS for method in requested):
        raise ValueError("Select at least one supported validation method.")
    directory = Path(args.output_dir) / "validations" / validation_id
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "status": "running", "validation_id": validation_id,
        "case_sha256": context["case_sha256"], "model_code_sha256": context["model_code_sha256"],
        "environment": runtime_environment(), "requested_methods": requested,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "budgets": context["metadata"]["budgets"],
        "provenance": collect_provenance(
            ROOT, [__file__, ROOT / "examples" / "_ckd_mixture_case.py",
                   ROOT / "tests" / "benchmark" / "benchmark_metrics.py",
                   ROOT / "tests" / "benchmark" / "diffgrid_nuts_storage.py"],
            {"validation_id": validation_id, "methods": requested}),
    }
    write_json(directory / "validation.json", report)
    started = time.perf_counter()
    try:
        metrics, arrays = evaluate_context(context)
        report.update(metrics)
        write_npz(directory / "spectra.npz", **arrays)
        report.update(status="completed", passed=all(report["methods"][method]["passed"] for method in requested),
                      spectra_sha256=sha256(directory / "spectra.npz"), elapsed_seconds=time.perf_counter() - started)
        report["residuals"] = {"filename": "spectra.npz", "sha256": report["spectra_sha256"]}
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
        write_json(directory / "validation.json", report)
        raise
    write_json(directory / "validation.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Save bundled line inputs, tables and one fixed mock observation")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--samples-per-band", type=int, default=1024)
    prepare.add_argument("--ng", type=int, default=16)
    prepare.add_argument("--temperature-nodes", type=int, default=21)
    prepare.add_argument("--validation-points", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=0)
    validate = commands.add_parser("validate", help="Save all four comparisons and gate the selected methods")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--validation-id", required=True)
    validate.add_argument("--methods", nargs="+", choices=METHODS, default=["lbl", "rorr"])
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        context = prepare_case(args.output_dir, samples_per_band=args.samples_per_band, ng=args.ng,
                               temperature_nodes=args.temperature_nodes, validation_points=args.validation_points, seed=args.seed)
        print(f"Prepared {args.output_dir}; case SHA256 {context['case_sha256']}")
        return 0
    report = validate_case(args)
    for method, row in report["methods"].items():
        maximum = max((check["max_error_in_noise"] for check in row["spectrum_checks"]
                       if check["max_error_in_noise"] is not None), default=float("nan"))
        print(f"{method}: passed={row['passed']}, max error/noise={maximum:.6g}, local gradients={row['gradient_passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
