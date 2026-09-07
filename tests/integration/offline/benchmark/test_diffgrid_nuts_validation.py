"""Small observation-space contracts for the manual DiffGrid benchmark."""

from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = ROOT / "tests" / "benchmark"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK_DIR))
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    module = importlib.import_module("diffgrid_nuts_benchmark")
    module._load_scientific_runtime()
    return module


@pytest.fixture
def validation(benchmark):
    return importlib.import_module("diffgrid_nuts_validation")


@pytest.fixture
def settings():
    return {
        "seed": 17,
        "num_prior_points": 1,
        "max_interpolation_error_in_noise": 0.01,
        "max_q": 0.1,
        "gradient_tolerance": 0.001,
        "steps": [1e-2, 1e-3, 1e-4, 1e-5],
    }


@pytest.fixture
def analytic_potential(validation, monkeypatch):
    """Keep the observation contracts runnable without optional NumPyro."""
    import jax.numpy as jnp

    monkeypatch.setattr(
        validation,
        "_potential_functions",
        lambda benchmark, forwards, *args: {
            method: lambda position: jnp.sum(position**2) for method in forwards
        },
    )


def _evaluate(validation, benchmark, fixture, settings):
    return validation.evaluate_case(
        benchmark,
        fixture.config,
        fixture.truth,
        fixture.bounds,
        fixture.case,
        fixture.context,
        fixture.teacher,
        fixture.diffgrid,
        settings,
    )


@pytest.fixture
def synthetic_case(benchmark):
    """Exercise the actual emission, rotation, LSF, and RV observation path."""
    import jax.numpy as jnp

    config = benchmark.CaseConfig(
        number_of_observed_wavenumbers=6,
        number_of_wavenumbers=64,
        number_of_layers=3,
        number_of_temperature_nodes=3,
        model_wavelength_margin=40.0,
        pressure_top=0.1,
        pressure_bottom=1.0,
    )
    nu_data, wavelengths = benchmark._observation_grid(config)
    nu_grid, model_wavelengths, resolution = benchmark._model_grid(wavelengths, config)
    art = benchmark.ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=config.pressure_top,
        pressure_btm=config.pressure_bottom,
        nlayer=config.number_of_layers,
    )
    art.change_temperature_range(config.temperature_min, config.temperature_max)
    spectral_shape = jnp.exp(-(((jnp.asarray(nu_grid) - np.mean(nu_grid)) / 8.0) ** 2))
    teacher = SimpleNamespace(
        method="analytic",
        ready=True,
        nu_grid=nu_grid,
        molmass=16.0,
        Tmin=config.temperature_min,
        Tmax=config.temperature_max,
        xsmatrix=lambda temperature, pressure: 1.0e-22
        * jnp.exp(-700.0 / temperature[:, None])
        * (1.0 + spectral_shape[None, :])
        * pressure[:, None] ** 0.1,
    )
    diffgrid = benchmark.OpaDiffgrid(
        teacher,
        np.array([1500.0, 800.0, 400.0]),
        np.asarray(art.pressure),
    )
    context = {
        "art": art,
        "opa_cia": SimpleNamespace(
            logacia_matrix=lambda temperature: jnp.broadcast_to(
                -46.0 + 0.1 * jnp.log10(temperature[:, None] / 1000.0),
                (config.number_of_layers, config.number_of_wavenumbers),
            )
        ),
        "nu_data": jnp.asarray(nu_data),
        "nu_grid": jnp.asarray(nu_grid),
        "velocity_array": benchmark.velocity_grid(resolution, config.maximum_vsini),
        "instrument_beta": benchmark.resolution_to_gaussian_std(
            config.instrument_resolution
        ),
        "hydrogen_volume_mixing_ratio": 0.85,
        "cia_temperature_grid": np.array([300.0, 2000.0]),
        "cia_wavenumber_grid": np.array([nu_grid[0] - 1.0, nu_grid[-1] + 1.0]),
    }
    truth = dict(benchmark.TRUTH)
    bounds = dict(benchmark.PRIOR_BOUNDS)
    forward = benchmark._make_forward_model(teacher, context, config)
    observed = np.asarray(
        forward(
            art.powerlaw_temperature(
                truth["temperature_at_1bar"], truth["temperature_index"]
            ),
            truth["methane_mass_mixing_ratio"],
            truth["radius"],
            truth["radial_velocity"],
            truth["vsini"],
        )
    )
    case = {
        "nu_grid": nu_grid,
        "nu_data": nu_data,
        "wavelength_data": wavelengths,
        "wavelength_grid": model_wavelengths,
        "model_resolution": np.array(resolution),
        "observed_flux": observed,
        "teacher_flux": observed.copy(),
        "diffgrid_flux": observed.copy(),
    }
    return SimpleNamespace(
        config=config,
        truth=truth,
        bounds=bounds,
        context=context,
        teacher=teacher,
        diffgrid=diffgrid,
        case=case,
    )


@pytest.fixture
def saved_case(benchmark, validation, synthetic_case, tmp_path, monkeypatch):
    storage = importlib.import_module("diffgrid_nuts_storage")
    paths = benchmark._case_paths(tmp_path)
    storage.write_npz(paths["case"], **synthetic_case.case)
    for name in ("premodit", "premodit_metadata", "diffgrid", "diffgrid_metadata"):
        paths[name].write_bytes(name.encode())
    cia = tmp_path / "synthetic.cia"
    cia.write_bytes(b"Synthetic CIA coefficients are supplied by the test fixture.")
    metadata = {
        "schema_version": 1,
        "config": asdict(synthetic_case.config),
        "truth": synthetic_case.truth,
        "prior_bounds": synthetic_case.bounds,
        "artifacts": {
            f"{name}_sha256": storage.sha256(path)
            for name, path in paths.items()
            if name != "prepare"
        },
        "inputs": {"cia_path": str(cia), "cia_sha256": storage.sha256(cia)},
        "environment": {"jax_enable_x64": True},
        "database_provenance": {
            "molecular_database": {
                "files": [{"sha256": storage.sha256(paths["premodit"])}]
            }
        },
    }
    storage.write_json(paths["prepare"], metadata)
    monkeypatch.setattr(
        benchmark,
        "_load_opacity",
        lambda method, *args, **kwargs: {
            "premodit": synthetic_case.teacher,
            "diffgrid": synthetic_case.diffgrid,
        }[method],
    )
    monkeypatch.setattr(
        benchmark, "_forward_context", lambda *args: synthetic_case.context
    )
    return SimpleNamespace(
        paths=paths,
        metadata=metadata,
        storage=storage,
        output_dir=tmp_path,
        synthetic=synthetic_case,
    )


def test_validate_cli_preserves_existing_precision_threshold(benchmark, tmp_path):
    parser = benchmark._parser()
    args = parser.parse_args(
        ["validate", "--output-dir", str(tmp_path), "--validation-id", "baseline"]
    )
    assert args.validation_id == "baseline"
    assert args.seed == 0 and args.num_prior_points == 16
    assert args.max_interpolation_error_in_noise == 0.01
    assert args.max_q == 0.1 and args.gradient_tolerance == 0.001
    assert not args.allow_code_revision
    assert parser.parse_args(["prepare"]).max_interpolation_error_in_noise == 0.01
    with pytest.raises(SystemExit):
        parser.parse_args(["validate"])


@pytest.mark.parametrize("field", ["observed_flux", "nu_data"])
def test_changed_observation_inputs_leave_failed_validation(
    benchmark, saved_case, field
):
    case = dict(saved_case.synthetic.case)
    case[field] = np.asarray(case[field]) + 0.01
    saved_case.storage.write_npz(saved_case.paths["case"], **case)
    args = benchmark._parser().parse_args(
        [
            "validate",
            "--output-dir",
            str(saved_case.output_dir),
            "--validation-id",
            "changed-input",
        ]
    )
    with pytest.raises(ValueError, match="digest|hash"):
        args.handler(args)
    saved = saved_case.output_dir / "validations/changed-input/validation.json"
    report = json.loads(saved.read_text())
    assert report["status"] == "failed"
    assert (
        "digest" in report["failure"]["message"]
        or "hash" in report["failure"]["message"]
    )
    assert not saved.with_name("residuals.npz").exists()
    original = saved.read_bytes()
    with pytest.raises(FileExistsError):
        args.handler(args)
    assert saved.read_bytes() == original


@pytest.mark.parametrize("identifier", ["../outside", "a/b", "a\\b", ".", ""])
def test_validation_id_cannot_escape_output_directory(benchmark, tmp_path, identifier):
    with pytest.raises((SystemExit, ValueError)):
        args = benchmark._parser().parse_args(
            [
                "validate",
                "--output-dir",
                str(tmp_path),
                "--validation-id",
                identifier,
            ]
        )
        args.handler(args)
    assert not (tmp_path / "validations").exists()


@pytest.mark.parametrize("real_potential", [False, True])
def test_exact_opacity_interpolant_passes_actual_observation_path(
    validation, benchmark, synthetic_case, settings, request, real_potential
):
    if real_potential:
        pytest.importorskip("numpyro")
    else:
        request.getfixturevalue("analytic_potential")
    report, arrays = _evaluate(validation, benchmark, synthetic_case, settings)
    assert report["passed"], report
    assert len(report["points"]) == 5 + 2 + settings["num_prior_points"]
    midpoint_points = [
        point for point in report["points"] if point["source"] == "opacity_midpoint"
    ]
    actual_midpoints = [point["temperature_profile"][0] for point in midpoint_points]
    np.testing.assert_allclose(sorted(actual_midpoints), [1600.0 / 3.0, 24000.0 / 23.0])
    assert (
        max(point["accuracy"]["max_error_in_noise"] for point in report["points"])
        < 1e-10
    )
    assert report["gradients"]
    for point in report["gradients"]:
        for method in point["methods"].values():
            assert all(
                method[name]["passed"]
                for name in ("forward", "log_likelihood", "potential")
            )
        assert point["between_methods"]["forward"]["scaled"] < 1e-8
    assert arrays and all(np.all(np.isfinite(array)) for array in arrays.values())


@pytest.mark.parametrize(
    "fault",
    ["nonfinite", "offset", "broken_gradient", "missing_numpyro", "nonfinite_domain"],
)
def test_scientific_failures_are_distinguished_from_completion(
    validation, benchmark, saved_case, settings, analytic_potential, monkeypatch, fault
):
    import jax
    import jax.numpy as jnp

    def factory(opacity, context, config):
        def forward(temperature, methane, radius, radial_velocity, vsini):
            value = jnp.mean(temperature) / 1000.0 + radius + methane
            result = jnp.full(context["nu_data"].shape, value)
            if opacity.method == "diffgrid":
                if fault == "nonfinite":
                    return result * jnp.nan
                if fault == "offset":
                    return result + config.noise_sigma
                if fault == "broken_gradient":
                    return jax.lax.stop_gradient(result)
            return result

        return forward

    monkeypatch.setattr(benchmark, "_make_forward_model", factory)
    if fault == "nonfinite_domain":
        saved_case.synthetic.context["cia_temperature_grid"] = np.array([300.0, np.nan])
    if fault == "missing_numpyro":

        def unavailable(*args):
            raise ModuleNotFoundError("NumPyro deliberately unavailable")

        monkeypatch.setattr(validation, "_potential_functions", unavailable)
    args = benchmark._parser().parse_args(
        [
            "validate",
            "--output-dir",
            str(saved_case.output_dir),
            "--validation-id",
            fault,
            "--num-prior-points",
            "1",
        ]
    )
    with pytest.raises(SystemExit) as error:
        args.handler(args)
    assert error.value.code == 1
    path = saved_case.output_dir / "validations" / fault / "validation.json"
    report = json.loads(
        path.read_text(),
        parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"),
    )
    assert report["status"] == "completed"
    assert report["passed"] is False
    if fault == "nonfinite_domain":
        assert report["coverage"]["checks"]["temperature_grids"] is False
        assert report["coverage"]["temperature_ranges"][2] == [None, None]
    elif fault == "missing_numpyro":
        assert report["potential_status"] == "unavailable"
        assert all(point["accuracy"]["passed"] for point in report["points"])
    elif fault == "broken_gradient":
        assert all(point["accuracy"]["passed"] for point in report["points"])
        assert any(
            not point["methods"]["diffgrid"]["forward"]["passed"]
            for point in report["gradients"]
        )
    else:
        assert any(not point["accuracy"]["passed"] for point in report["points"])
    with np.load(path.with_name("residuals.npz"), allow_pickle=False) as residuals:
        assert residuals.files
    with pytest.raises(ValueError, match="validation|Validation|passed"):
        benchmark._validation_gate(
            saved_case.output_dir,
            saved_case.storage.sha256(saved_case.paths["prepare"]),
            validation_id=fault,
        )


def test_numpyro_potential_includes_uniform_transform_jacobian(
    validation, benchmark, synthetic_case
):
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp

    fixture = synthetic_case
    observation = np.zeros_like(fixture.case["observed_flux"])
    forwards = {
        method: lambda *args: jnp.zeros_like(observation)
        for method in ("premodit", "diffgrid")
    }
    functions = validation._potential_functions(
        benchmark,
        forwards,
        fixture.context,
        fixture.config,
        fixture.bounds,
        fixture.truth,
        observation,
        0,
    )
    center = jnp.zeros(len(fixture.truth))
    expected = (
        0.5 * observation.size * np.log(2 * np.pi * fixture.config.noise_sigma**2)
    )
    expected += len(fixture.truth) * np.log(4.0)
    position = jnp.linspace(-1.0, 1.0, len(fixture.truth))
    for potential in functions.values():
        np.testing.assert_allclose(potential(center), expected, atol=1e-12)
        np.testing.assert_allclose(
            jax.grad(potential)(position), jnp.tanh(position / 2), atol=1e-12
        )


def test_validation_selection_and_saved_residuals_are_verified(
    benchmark, validation, saved_case, monkeypatch
):
    digest = saved_case.storage.sha256(saved_case.paths["prepare"])
    legacy = {}
    benchmark._validate_result_evidence(saved_case.output_dir, legacy, digest, None)
    assert legacy["accuracy_validation"] == {"status": "not_run", "passed": None}
    monkeypatch.setattr(
        validation,
        "evaluate_case",
        lambda *args, **kwargs: (
            {
                "passed": True,
                "coverage": {"passed": True},
                "points": [],
                "gradients": [],
                "reference_convergence": {"status": "not_established"},
            },
            {"residual_in_noise": np.zeros((1, 6))},
        ),
    )
    args = benchmark._parser().parse_args(
        [
            "validate",
            "--output-dir",
            str(saved_case.output_dir),
            "--validation-id",
            "baseline",
        ]
    )
    args.handler(args)
    digest = saved_case.storage.sha256(saved_case.paths["prepare"])
    with pytest.raises(ValueError, match="validation|Validation"):
        benchmark._validation_gate(saved_case.output_dir, digest)
    selected = benchmark._validation_gate(
        saved_case.output_dir, digest, validation_id="baseline"
    )
    assert selected["passed"]
    with pytest.raises(ValueError, match="code|Code|revision"):
        benchmark._validation_gate(
            saved_case.output_dir,
            digest,
            code_sha256="wrong-code",
            validation_id="baseline",
        )
    report_path = saved_case.output_dir / "validations/baseline/validation.json"
    report = json.loads(report_path.read_text())
    with pytest.raises(ValueError, match="provenance"):
        benchmark._validate_result_evidence(
            saved_case.output_dir, {}, digest, "baseline"
        )
    changed_environment = dict(report["environment"], jax_enable_x64=False)
    with pytest.raises(ValueError, match="environment"):
        benchmark._validation_gate(
            saved_case.output_dir,
            digest,
            validation_id="baseline",
            environment=changed_environment,
        )
    recorded_run = {
        "validation": selected,
        "provenance": report["provenance"],
        "environment": report["environment"],
    }
    report["scope"] = "Changed after the run recorded its evidence."
    saved_case.storage.write_json(report_path, report)
    with pytest.raises(ValueError, match="report digest"):
        benchmark._validate_result_evidence(
            saved_case.output_dir, recorded_run, digest, None
        )
    residuals = saved_case.output_dir / "validations/baseline/residuals.npz"
    saved_case.storage.write_npz(residuals, residual_in_noise=np.ones((1, 6)))
    with pytest.raises(ValueError, match="digest|hash|residual"):
        benchmark._validation_gate(
            saved_case.output_dir, digest, validation_id="baseline"
        )


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("cia_temperature", "prior_temperature"),
        ("cia_wavenumber", "cia_wavenumbers"),
        ("edge_coverage", "observation_padding"),
        ("pressure", "pressure_grid"),
        ("nonfinite_temperature", "temperature_grids"),
        ("short_rotation_kernel", "rotation_support"),
    ],
)
def test_domain_checks_reject_uncovered_layers_and_observation_edges(
    validation, synthetic_case, fault, failed_check, monkeypatch
):
    fixture = synthetic_case
    context = dict(fixture.context)
    case = dict(fixture.case)
    if fault == "cia_temperature":
        context["cia_temperature_grid"] = np.array([900.0, 1000.0])
    elif fault == "cia_wavenumber":
        context["cia_wavenumber_grid"] = case["nu_grid"][1:-1]
    elif fault == "edge_coverage":
        case["nu_data"] = case["nu_grid"][[0, -1]]
    elif fault == "nonfinite_temperature":
        context["cia_temperature_grid"] = np.array([300.0, np.nan])
    elif fault == "short_rotation_kernel":
        context["velocity_array"] = np.array([-10.0, 0.0, 10.0])
    else:
        monkeypatch.setattr(
            fixture.diffgrid, "pressure_grid", fixture.diffgrid.pressure_grid * 2
        )
    report = validation._coverage(
        fixture.config, fixture.bounds, case, context, fixture.teacher, fixture.diffgrid
    )
    assert report["passed"] is False
    assert report["checks"][failed_check] is False
    if fault == "cia_temperature":
        assert report["invalid_temperature_layers"]
        assert report["common_temperature_range"] == [900.0, 1000.0]


@pytest.mark.parametrize(
    ("error_in_noise", "expected_status"), [(0.0005, "passed"), (0.002, "failed")]
)
def test_reference_refinement_uses_one_tenth_of_observation_budget(
    validation,
    benchmark,
    saved_case,
    settings,
    error_in_noise,
    expected_status,
    monkeypatch,
):
    baseline = json.loads(saved_case.paths["prepare"].read_text())
    refined = json.loads(json.dumps(baseline))
    refined["config"]["broadening_resolution"] /= 2
    case = dict(saved_case.synthetic.case)
    case["teacher_flux"] = np.full_like(case["teacher_flux"], 1e6)

    def factory(opacity, context, config):
        delta = (
            error_in_noise * config.noise_sigma
            if config.broadening_resolution
            < saved_case.synthetic.config.broadening_resolution
            else 0.0
        )
        return lambda *args: saved_case.synthetic.case["teacher_flux"] + delta

    monkeypatch.setattr(benchmark, "_make_forward_model", factory)
    saved_case.storage.write_npz(saved_case.paths["case"], **case)
    refined["artifacts"]["case_sha256"] = saved_case.storage.sha256(
        saved_case.paths["case"]
    )
    saved_case.storage.write_json(saved_case.paths["prepare"], refined)
    report = validation._reference_check(
        benchmark,
        saved_case.output_dir,
        baseline,
        saved_case.synthetic.case,
        saved_case.synthetic.context,
        saved_case.synthetic.config,
        settings,
        False,
        saved_case.synthetic.teacher,
    )
    assert report["status"] == expected_status
    np.testing.assert_allclose(report["metrics"]["max_error_in_noise"], error_in_noise)
    assert "not an absolute-accuracy proof" in report["scope"]


def test_reference_with_changed_observation_grid_is_rejected(
    validation, benchmark, saved_case, settings
):
    baseline = json.loads(saved_case.paths["prepare"].read_text())
    case = dict(saved_case.synthetic.case)
    case["nu_data"] = case["nu_data"] + 0.01
    saved_case.storage.write_npz(saved_case.paths["case"], **case)
    changed = json.loads(json.dumps(baseline))
    changed["artifacts"]["case_sha256"] = saved_case.storage.sha256(
        saved_case.paths["case"]
    )
    saved_case.storage.write_json(saved_case.paths["prepare"], changed)
    with pytest.raises(ValueError, match="observation grid"):
        validation._reference_check(
            benchmark,
            saved_case.output_dir,
            baseline,
            saved_case.synthetic.case,
            saved_case.synthetic.context,
            saved_case.synthetic.config,
            settings,
            False,
            saved_case.synthetic.teacher,
        )


@pytest.mark.parametrize("fault", ["coarser", "unchanged", "pressure", "precision"])
def test_reference_rejects_coarsening_and_changed_physics(
    validation, benchmark, saved_case, settings, fault, monkeypatch
):
    baseline = json.loads(saved_case.paths["prepare"].read_text())
    refined = json.loads(json.dumps(baseline))
    refined["config"]["broadening_resolution"] /= 2
    if fault == "coarser":
        refined["config"]["number_of_wavenumbers"] //= 2
    elif fault == "unchanged":
        refined = baseline
    elif fault == "precision":
        refined["environment"]["jax_enable_x64"] = False
    else:
        context = dict(saved_case.synthetic.context)
        context["art"] = SimpleNamespace(
            pressure_boundary=np.asarray(context["art"].pressure_boundary) * 2
        )
        monkeypatch.setattr(benchmark, "_forward_context", lambda *args: context)
    saved_case.storage.write_json(saved_case.paths["prepare"], refined)
    with pytest.raises(ValueError, match="refine|boundaries|precision"):
        validation._reference_check(
            benchmark,
            saved_case.output_dir,
            baseline,
            saved_case.synthetic.case,
            saved_case.synthetic.context,
            saved_case.synthetic.config,
            settings,
            False,
            saved_case.synthetic.teacher,
        )


def test_optional_reference_runtime_failure_cannot_retain_success(
    benchmark, validation, saved_case, monkeypatch
):
    monkeypatch.setattr(
        validation,
        "evaluate_case",
        lambda *args, **kwargs: (
            {"passed": True},
            {"residual_in_noise": np.zeros((1, 6))},
        ),
    )

    def fail_reference(*args):
        raise ValueError("Synthetic reference input mismatch")

    monkeypatch.setattr(validation, "_reference_check", fail_reference)
    args = benchmark._parser().parse_args(
        [
            "validate",
            "--output-dir",
            str(saved_case.output_dir),
            "--validation-id",
            "reference-failure",
            "--reference-output-dir",
            str(saved_case.output_dir / "refined"),
        ]
    )
    with pytest.raises(ValueError, match="reference input mismatch"):
        args.handler(args)
    path = saved_case.output_dir / "validations/reference-failure/validation.json"
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["passed"] is False
    assert report["failure"]["stage"] == "reference_refinement"
    assert not path.with_name("residuals.npz").exists()
