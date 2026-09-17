"""Offline-capable validation of the existing DiffGrid NUTS benchmark case.

Finite probes establish agreement with a saved teacher at those probes only.
They do not establish accuracy over the entire prior or convergence of the
production PreMODIT reference. Numerical settings are recorded before evaluation.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark_metrics import (
    derivative_difference,
    directional_check,
    observation_error,
)
from diffgrid_nuts_storage import sha256, validate_run_id, write_npz


DEFAULT_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def _json_report(value):
    """Keep failed numerical values JSON-safe; residual archives preserve NaNs."""
    if isinstance(value, dict):
        return {key: _json_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_report(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _clip(temperature, lower, upper):
    raw = np.asarray(temperature)
    region = np.where(raw < lower, -1, np.where(raw > upper, 1, 0))
    return {
        "counts": {
            "below": int(np.count_nonzero(region == -1)),
            "above": int(np.count_nonzero(region == 1)),
        },
        "below": np.flatnonzero(region == -1).tolist(),
        "above": np.flatnonzero(region == 1).tolist(),
        "at_boundary": np.flatnonzero(
            np.isclose(raw, lower, rtol=1e-12, atol=0)
            | np.isclose(raw, upper, rtol=1e-12, atol=0)
        ).tolist(),
        "region": region.tolist(),
    }


def _coverage(case_config, prior_bounds, case, context, teacher, diffgrid):
    """Check the full prior's temperature envelope and finite observation kernel."""
    from exojax.utils.constants import c

    pressure = np.asarray(context["art"].pressure)
    nu = np.asarray(case["nu_grid"])
    observed_nu = np.asarray(case["nu_data"])
    cia_t = np.asarray(context["cia_temperature_grid"])
    cia_nu = np.asarray(context["cia_wavenumber_grid"])
    temperature_limits = [
        [
            float(np.min(diffgrid.temperature_grid)),
            float(np.max(diffgrid.temperature_grid)),
        ],
        [float(teacher.Tmin), float(teacher.Tmax)],
        [float(np.min(cia_t)), float(np.max(cia_t))],
    ]
    if hasattr(teacher, "T_gQT"):
        temperature_limits.append(
            [float(np.min(teacher.T_gQT)), float(np.max(teacher.T_gQT))]
        )
    lower = max(bounds[0] for bounds in temperature_limits)
    upper = min(bounds[1] for bounds in temperature_limits)
    envelope = np.asarray(
        [
            np.clip(
                t0 * pressure**alpha,
                case_config.temperature_min,
                case_config.temperature_max,
            )
            for t0 in prior_bounds["temperature_at_1bar"]
            for alpha in prior_bounds["temperature_index"]
        ]
    )
    velocities = np.asarray(context["velocity_array"])
    center = len(velocities) // 2
    rotation_indices = np.flatnonzero(np.abs(velocities) < prior_bounds["vsini"][1])
    rotation_radius = (
        int(np.max(np.abs(rotation_indices - center))) if rotation_indices.size else 0
    )
    # Gaussian sampling uses this entire finite kernel. Do not truncate its tails
    # just to accept a narrow model wavelength margin.
    padding = center + rotation_radius
    usable = (
        [float(nu[padding]), float(nu[-padding - 1])] if 2 * padding < len(nu) else None
    )
    shifted = [
        float(np.min(observed_nu) * (1 + prior_bounds["radial_velocity"][0] / c)),
        float(np.max(observed_nu) * (1 + prior_bounds["radial_velocity"][1] / c)),
    ]
    checks = {
        "positive_pressure": bool(
            np.all(np.isfinite(pressure)) and np.all(pressure > 0)
        ),
        "pressure_grid": bool(
            np.array_equal(pressure, np.asarray(diffgrid.pressure_grid))
        ),
        "opacity_wavenumbers": bool(
            np.array_equal(nu, np.asarray(teacher.nu_grid))
            and np.array_equal(nu, np.asarray(diffgrid.nu_grid))
        ),
        "ordered_wavenumbers": bool(
            np.all(np.isfinite(nu))
            and np.all(nu > 0)
            and np.all(np.isfinite(observed_nu))
            and np.all(observed_nu > 0)
            and np.all(np.diff(nu) > 0)
            and np.all(np.diff(observed_nu) > 0)
        ),
        "temperature_grids": bool(
            np.all(np.isfinite(temperature_limits))
            and np.all(np.asarray(temperature_limits) > 0)
            and np.all(np.isfinite(cia_t))
            and np.all(np.diff(cia_t) > 0)
        ),
        "rotation_support": bool(
            np.all(np.isfinite(velocities))
            and prior_bounds["vsini"][0] > 0
            and np.min(velocities) <= -prior_bounds["vsini"][1]
            and np.max(velocities) >= prior_bounds["vsini"][1]
        ),
        "cia_wavenumbers": bool(
            np.all(np.isfinite(cia_nu))
            and np.all(np.diff(cia_nu) > 0)
            and np.min(cia_nu) <= np.min(nu)
            and np.max(cia_nu) >= np.max(nu)
        ),
        "prior_temperature": bool(
            lower <= upper
            and np.all(np.isfinite(envelope))
            and np.all(envelope >= lower)
            and np.all(envelope <= upper)
        ),
        "observation_padding": bool(
            usable is not None and shifted[0] >= usable[0] and shifted[1] <= usable[1]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "common_temperature_range": [lower, upper],
        "temperature_ranges": temperature_limits,
        "pressure_profile": pressure.tolist(),
        "prior_temperature_envelope": [
            np.min(envelope, axis=0).tolist(),
            np.max(envelope, axis=0).tolist(),
        ],
        "invalid_temperature_layers": np.argwhere(
            (envelope < lower) | (envelope > upper) | ~np.isfinite(envelope)
        ).tolist(),
        "observation": {
            "padding_grid_points": padding,
            "usable_wavenumbers": usable,
            "rv_shifted_wavenumbers": shifted,
            "operator": "rigid rotation, full finite Gaussian kernel, RV, point sampling; no bins",
        },
    }


def _potential_functions(
    benchmark, forwards, context, case_config, prior_bounds, truth, observation, seed
):
    """Use NumPyro's actual unconstrained potential, including its Jacobian."""
    import jax
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    functions = {}
    names = list(benchmark.TRUTH)
    for method, forward in forwards.items():
        model = benchmark._make_numpyro_model(
            forward, context["art"], case_config, prior_bounds
        )
        info = initialize_model(
            jax.random.PRNGKey(seed),
            model,
            init_strategy=init_to_value(values=truth),
            model_kwargs={"observation": observation},
            validate_grad=False,
        )

        def potential(q, potential_fn=info.potential_fn):
            return potential_fn({name: q[index] for index, name in enumerate(names)})

        functions[method] = jax.jit(potential)
    return functions


def evaluate_case(
    benchmark,
    case_config,
    truth,
    prior_bounds,
    case,
    context,
    teacher,
    diffgrid,
    settings,
    checkpoint=None,
):
    """Return a JSON report and arrays for one immutable prepared observation."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import expit
    from jax.scipy.stats import norm
    from exojax.utils.constants import c
    from exojax.opacity.diffgrid.diagnostics import (
        diffgrid_interval_midpoint_temperatures,
    )

    names = list(benchmark.TRUTH)
    lower = np.asarray([prior_bounds[name][0] for name in names])
    width = np.asarray([prior_bounds[name][1] for name in names]) - lower
    if not np.all(np.isfinite(width)) or np.any(width <= 0):
        raise ValueError("Prior bounds must be finite and strictly increasing.")
    if np.asarray(case["observed_flux"]).shape != np.asarray(case["nu_data"]).shape:
        raise ValueError("Observation and sampling grid shapes differ.")
    observation = jnp.asarray(case["observed_flux"])
    if not np.all(np.isfinite(observation)):
        raise ValueError("Saved observation contains nonfinite values.")
    pressure = np.asarray(context["art"].pressure)
    t_index, a_index = (
        names.index("temperature_at_1bar"),
        names.index("temperature_index"),
    )
    coverage = _coverage(case_config, prior_bounds, case, context, teacher, diffgrid)
    forwards = {
        method: benchmark._make_forward_model(opacity, context, case_config)
        for method, opacity in (("premodit", teacher), ("diffgrid", diffgrid))
    }

    def raw_temperature(unit):
        physical = lower + width * np.asarray(unit)
        return physical[t_index] * pressure ** physical[a_index]

    rv_index = names.index("radial_velocity")
    physical_coordinates = jax.jit(lambda unit: lower + width * unit)

    @jax.jit
    def sampling_coordinates(unit):
        # Match the current observation operator, including its Doppler convention.
        physical = lower + width * unit
        return context["nu_data"] * (1.0 + physical[rv_index] / c)

    def region(unit):
        clipping = _clip(
            raw_temperature(unit),
            case_config.temperature_min,
            case_config.temperature_max,
        )["region"]
        shifted = np.asarray(sampling_coordinates(unit))
        nu_grid = np.asarray(context["nu_grid"])
        # Exact knots have their own labels, distinct from either open cell.
        cells = np.searchsorted(nu_grid, shifted, side="left") + np.searchsorted(
            nu_grid, shifted, side="right"
        )
        return np.concatenate((clipping, cells))

    def in_domain(unit):
        unit = np.asarray(unit)
        temperatures = np.clip(
            raw_temperature(unit),
            case_config.temperature_min,
            case_config.temperature_max,
        )
        limits = coverage["common_temperature_range"]
        return bool(
            np.all(np.isfinite(unit))
            and np.all(unit > 0)
            and np.all(unit < 1)
            and np.all(temperatures >= limits[0])
            and np.all(temperatures <= limits[1])
        )

    def predictor(forward):
        def predict(unit):
            p = lower + width * unit
            values = dict(zip(names, p))
            temperature = context["art"].powerlaw_temperature(
                values["temperature_at_1bar"], values["temperature_index"]
            )
            return forward(
                temperature,
                values["methane_mass_mixing_ratio"],
                values["radius"],
                values["radial_velocity"],
                values["vsini"],
            )

        return jax.jit(predict)

    predictors = {method: predictor(forward) for method, forward in forwards.items()}
    likelihoods = {
        method: jax.jit(
            lambda unit, predict=predict: norm.logpdf(
                observation, predict(unit), case_config.noise_sigma
            ).sum()
        )
        for method, predict in predictors.items()
    }
    probes = []
    for label, t0, alpha in benchmark._validation_profiles(truth, prior_bounds):
        values = dict(truth, temperature_at_1bar=t0, temperature_index=alpha)
        probes.append((label, "legacy_profile", values, None))
    for index, temperature in enumerate(
        diffgrid_interval_midpoint_temperatures(diffgrid)
    ):
        probes.append(
            (
                f"inverse_temperature_midpoint_{index}",
                "opacity_midpoint",
                dict(truth),
                np.full(pressure.shape, temperature),
            )
        )
    rng = np.random.default_rng(settings["seed"])
    for index in range(settings["num_prior_points"]):
        values = dict(zip(names, lower + width * rng.uniform(size=len(names))))
        probes.append((f"prior_interior_{index}", "prior_interior", values, None))
    report = {
        "passed": False,
        "coverage": coverage,
        "points": [],
        "gradients": [],
        "parameter_order": names,
        "reference_convergence": {
            "status": "not_established",
            "reason": "The prepared PreMODIT teacher has not been independently refined for this case.",
        },
        "scope": "Finite probes relative to the saved teacher; no guarantee over the entire prior or of absolute accuracy.",
        "coordinates": {
            "forward_and_likelihood": "unit prior coordinates: theta = lower + prior_width * unit",
            "potential": "NumPyro unconstrained q (Uniform-prior log odds)",
            "potential_definition": "U(q) = -log L(theta(q)) - log p(theta(q)) - log|dtheta/dq|",
            "derivative_scale": "max absolute AD-FD / max(1, max absolute AD, max absolute FD); forward divided by noise_sigma",
            "smooth_regions": "Temperature clipping states and linear RV sampling cells; exact RV knots have separate labels.",
        },
    }
    fluxes = {method: [] for method in forwards}
    temperatures_saved = []
    for label, source, values, override in probes:
        unit = (np.asarray([values[name] for name in names]) - lower) / width
        raw = raw_temperature(unit) if override is None else override
        temperature = np.clip(
            raw, case_config.temperature_min, case_config.temperature_max
        )
        temperatures_saved.append(temperature)
        limits = coverage["common_temperature_range"]
        valid = bool(
            coverage["passed"]
            and np.all(temperature >= limits[0])
            and np.all(temperature <= limits[1])
        )
        point = {
            "label": label,
            "source": source,
            "parameters": values,
            "temperature_profile": temperature.tolist(),
            "clip": _clip(
                raw, case_config.temperature_min, case_config.temperature_max
            ),
            "domain_valid": valid,
            "isothermal_override": override is not None,
        }
        for method, forward in forwards.items():
            if valid:
                flux = np.asarray(
                    forward(
                        jnp.asarray(temperature),
                        values["methane_mass_mixing_ratio"],
                        values["radius"],
                        values["radial_velocity"],
                        values["vsini"],
                    )
                )
            else:
                flux = np.full(observation.shape, np.nan)
            fluxes[method].append(flux)
        point["accuracy"] = observation_error(
            fluxes["diffgrid"][-1],
            fluxes["premodit"][-1],
            case_config.noise_sigma,
            max_error=settings["max_interpolation_error_in_noise"],
            max_q=settings["max_q"],
        )
        report["points"].append(point)
        if checkpoint:
            checkpoint(_json_report(report))
    try:
        potentials = (
            _potential_functions(
                benchmark,
                forwards,
                context,
                case_config,
                prior_bounds,
                truth,
                observation,
                settings["seed"],
            )
            if coverage["passed"]
            else {}
        )
        report["potential_status"] = "available" if potentials else "invalid_coverage"
    except ImportError as error:
        potentials = {}
        report["potential_status"] = "unavailable"
        report["potential_reason"] = str(error)
    if coverage["passed"]:
        for label, source, values, override in probes:
            if override is not None or (
                source != "prior_interior" and label != "mock parameters"
            ):
                continue
            unit = (np.asarray([values[name] for name in names]) - lower) / width
            direction = rng.normal(size=len(names))
            direction /= np.linalg.norm(direction)

            def stencil_resolved(center, plus, minus):
                active = direction != 0
                physical = np.asarray(physical_coordinates(center))
                shifted = np.asarray(sampling_coordinates(center))
                for endpoint in (plus, minus):
                    if np.any(
                        (np.asarray(physical_coordinates(endpoint)) == physical)[active]
                    ):
                        return False
                    if active[rv_index] and np.any(
                        np.asarray(sampling_coordinates(endpoint)) == shifted
                    ):
                        return False
                return True

            q = np.log(unit) - np.log1p(-unit)
            entry = {
                "label": label,
                "direction": direction.tolist(),
                "unit_position": unit.tolist(),
                "unconstrained_position": q.tolist(),
                "methods": {},
                "between_methods": {},
            }
            for method, predict in predictors.items():
                functions = {
                    "forward": lambda u, predict=predict: predict(u)
                    / case_config.noise_sigma,
                    "log_likelihood": likelihoods[method],
                }
                result = {}
                for name, function in functions.items():
                    result[name] = directional_check(
                        function,
                        unit,
                        direction,
                        steps=settings["steps"],
                        tolerance=settings["gradient_tolerance"],
                        in_domain=in_domain,
                        region=region,
                        stencil_resolved=stencil_resolved,
                    )
                if method in potentials:
                    result["potential"] = directional_check(
                        potentials[method],
                        q,
                        direction,
                        steps=settings["steps"],
                        tolerance=settings["gradient_tolerance"],
                        in_domain=lambda x: in_domain(np.asarray(expit(x))),
                        region=lambda x: region(np.asarray(expit(x))),
                        stencil_resolved=lambda x, plus, minus: stencil_resolved(
                            np.asarray(expit(x)),
                            np.asarray(expit(plus)),
                            np.asarray(expit(minus)),
                        ),
                    )
                else:
                    result["potential"] = {
                        "passed": False,
                        "reason": report["potential_status"],
                    }
                entry["methods"][method] = result
            for name in ("forward", "log_likelihood", "potential"):
                ad = [
                    entry["methods"][method][name].get("ad_directional_derivative")
                    for method in forwards
                ]
                if all(value is not None for value in ad):
                    entry["between_methods"][name] = derivative_difference(*ad)
            report["gradients"].append(entry)
            if checkpoint:
                checkpoint(_json_report(report))
    report["passed"] = bool(
        coverage["passed"]
        and all(point["accuracy"]["passed"] for point in report["points"])
        and report["gradients"]
        and all(
            check["passed"]
            for entry in report["gradients"]
            for method in entry["methods"].values()
            for check in method.values()
        )
    )
    arrays = {method: np.asarray(values) for method, values in fluxes.items()}
    arrays.update(
        temperature_profiles=np.asarray(temperatures_saved),
        nu_data=np.asarray(case["nu_data"]),
        residual_in_noise=(arrays["diffgrid"] - arrays["premodit"])
        / case_config.noise_sigma,
    )
    return _json_report(report), arrays


def _reference_check(
    benchmark,
    reference_dir,
    base_metadata,
    base_case,
    base_context,
    case_config,
    settings,
    allow_code_revision,
    base_teacher,
):
    """Compare a separately prepared teacher without treating it as an exact solution."""
    paths, metadata, archive, _ = benchmark._load_case(reference_dir)
    with archive:
        case = dict(archive)
    benchmark._validate_artifacts(paths, metadata, ["premodit"])
    config = benchmark.CaseConfig(**metadata["config"])
    allowed = {
        "number_of_wavenumbers",
        "number_of_layers",
        "number_of_temperature_nodes",
        "premodit_diffmode",
        "broadening_resolution",
        "pressure_top",
        "pressure_bottom",
    }
    for key, value in asdict(case_config).items():
        if key not in allowed and getattr(config, key) != value:
            raise ValueError(
                f"Reference physical/observation configuration differs: {key}"
            )
    for key in ("truth", "prior_bounds", "database_provenance"):
        if metadata.get(key) != base_metadata.get(key):
            raise ValueError(f"Reference input differs: {key}")
    if metadata["inputs"]["cia_sha256"] != base_metadata["inputs"][
        "cia_sha256"
    ] or not np.array_equal(case["nu_data"], base_case["nu_data"]):
        raise ValueError("Reference CIA or observation grid differs.")
    source_files = (
        (metadata.get("database_provenance") or {})
        .get("molecular_database", {})
        .get("files", [])
    )
    if not any(item.get("sha256") for item in source_files):
        raise ValueError(
            "Reference refinement requires recorded database source hashes."
        )
    if (
        metadata["environment"]["jax_enable_x64"]
        != base_metadata["environment"]["jax_enable_x64"]
    ):
        raise ValueError("Reference and baseline prepared precision differs.")
    context = benchmark._forward_context(
        case["nu_grid"],
        base_case["nu_data"],
        float(case["model_resolution"]),
        config,
        Path(metadata["inputs"]["cia_path"]),
    )
    if not np.allclose(
        np.asarray(context["art"].pressure_boundary)[[0, -1]],
        np.asarray(base_context["art"].pressure_boundary)[[0, -1]],
        rtol=1e-12,
        atol=0,
    ):
        raise ValueError("Reference and baseline physical pressure boundaries differ.")
    refinements = [
        getattr(config, key) - getattr(case_config, key)
        for key in ("number_of_wavenumbers", "number_of_layers", "premodit_diffmode")
    ] + [case_config.broadening_resolution - config.broadening_resolution]
    if any(change < 0 for change in refinements) or not any(
        change > 0 for change in refinements
    ):
        raise ValueError(
            "Reference must refine numerical settings without coarsening others."
        )
    teacher = benchmark._load_opacity(
        "premodit", paths["premodit"], allow_code_revision
    )
    # Reevaluate both archives through the same current runtime. Saved forward
    # outputs may come from different code revisions or noisy observations.
    predictions = []
    truth = base_metadata["truth"]
    for opacity, model_context, model_config in (
        (base_teacher, base_context, case_config),
        (teacher, context, config),
    ):
        forward = benchmark._make_forward_model(opacity, model_context, model_config)
        temperature = model_context["art"].powerlaw_temperature(
            truth["temperature_at_1bar"], truth["temperature_index"]
        )
        predictions.append(
            np.asarray(
                forward(
                    temperature,
                    truth["methane_mass_mixing_ratio"],
                    truth["radius"],
                    truth["radial_velocity"],
                    truth["vsini"],
                )
            )
        )
    error = observation_error(
        predictions[1],
        predictions[0],
        case_config.noise_sigma,
        max_error=settings["max_interpolation_error_in_noise"] / 10,
        max_q=settings["max_q"] / 100,
    )
    return {
        "status": "passed" if error["passed"] else "failed",
        "scope": "One refinement at truth only; not an absolute-accuracy proof.",
        "evaluation": "Both saved teachers reevaluated in the validation runtime.",
        "metrics": error,
        "budget": {
            "max_error_in_noise": settings["max_interpolation_error_in_noise"] / 10,
            "max_q": settings["max_q"] / 100,
        },
        "config": asdict(config),
        "prepare_sha256": sha256(paths["prepare"]),
        "artifacts": metadata["artifacts"],
    }


def validate_case(args, benchmark):
    output_dir = args.output_dir.resolve()
    directory = output_dir / "validations" / validate_run_id(args.validation_id)
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "validation.json"
    state = {
        "passed": False,
        "validation_id": args.validation_id,
        "prepare_sha256": sha256(output_dir / "prepare.json")
        if (output_dir / "prepare.json").is_file()
        else None,
        "settings": {
            "seed": args.seed,
            "num_prior_points": args.num_prior_points,
            "max_interpolation_error_in_noise": args.max_interpolation_error_in_noise,
            "max_q": args.max_q,
            "gradient_tolerance": args.gradient_tolerance,
            "steps": list(DEFAULT_STEPS),
        },
    }
    with benchmark._record_execution(result_path, state):
        benchmark.config.update("jax_enable_x64", True)
        benchmark._load_scientific_runtime()
        state["provenance"] = benchmark._provenance(args, [output_dir / "prepare.json"])
        state["environment"] = benchmark._environment()
        benchmark._stage(state, result_path, "input_validation")
        paths, metadata, archive, digest = benchmark._load_case(output_dir)
        with archive:
            case = dict(archive)
        benchmark._validate_artifacts(paths, metadata, ["premodit", "diffgrid"])
        state["case_sha256"] = digest
        config = benchmark.CaseConfig(**metadata["config"])
        context = benchmark._forward_context(
            case["nu_grid"],
            case["nu_data"],
            float(case["model_resolution"]),
            config,
            Path(metadata["inputs"]["cia_path"]),
        )
        opacities = {
            method: benchmark._load_opacity(
                method, paths[method], args.allow_code_revision
            )
            for method in ("premodit", "diffgrid")
        }
        benchmark._stage(state, result_path, "accuracy_and_gradients")

        def checkpoint(report):
            state.update(report)
            benchmark._write_json(result_path, state)

        report, arrays = evaluate_case(
            benchmark,
            config,
            metadata["truth"],
            metadata["prior_bounds"],
            case,
            context,
            opacities["premodit"],
            opacities["diffgrid"],
            state["settings"],
            checkpoint,
        )
        state.update(report)
        passed = state["passed"]
        state["passed"] = False
        if args.reference_output_dir is not None:
            benchmark._stage(state, result_path, "reference_refinement")
            state["reference_convergence"] = _reference_check(
                benchmark,
                args.reference_output_dir.resolve(),
                metadata,
                case,
                context,
                config,
                state["settings"],
                args.allow_code_revision,
                opacities["premodit"],
            )
            passed = passed and state["reference_convergence"]["status"] == "passed"
        benchmark._stage(state, result_path, "residual_save")
        write_npz(directory / "residuals.npz", **arrays)
        state["residuals"] = {
            "filename": "residuals.npz",
            "sha256": sha256(directory / "residuals.npz"),
            "shapes": {name: list(value.shape) for name, value in arrays.items()},
        }
        state["passed"] = passed
    print(f"Validation {'passed' if state['passed'] else 'failed'}: {result_path}")
    if not state["passed"]:
        raise SystemExit(1)
