"""Offline contracts for the two-instrument CO observation example."""

import builtins
import copy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "examples" / "joint_observation_retrieval.py"


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("_joint_observation")


@pytest.fixture
def joint(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("joint_observation_retrieval")


@pytest.fixture
def synthetic_context(model, monkeypatch):
    """Isolate the actual LSF/RV/bin operators from expensive opacity setup."""
    import jax.numpy as jnp
    from exojax.postproc.specop import SopRotation

    grid = np.geomspace(4330.0, 4350.0, 1024)
    context = {"nu_grid": grid, "sop_rot": SopRotation(grid, vsini_max=20.0)}

    def rotated_flux(T0, alpha, MMR, gravity, vsini):
        coordinate = jnp.asarray(grid)
        return 1.0e4 + T0 * (
            jnp.exp(-jnp.square((coordinate - 4340.0) / 0.025))
            + 0.3 * jnp.sin((coordinate - 4340.0) * 30.0)
        )

    monkeypatch.setattr(model, "make_rotated_flux", lambda _: rotated_flux)
    return context


@pytest.fixture
def synthetic_case(model, synthetic_context):
    arrays = model.make_geometry(synthetic_context)
    forward = model.make_forward(synthetic_context, arrays)
    parameters = model.mock_truth()
    parameters.update(T0=1250.0, RV=39.0, offset_b=175.0, scale_a=0.8, scale_b=1.3)
    predictions = forward(parameters)
    for name in ("a", "b"):
        arrays[f"observed_{name}"] = np.asarray(predictions[name]) + np.linspace(
            -300.0, 400.0, len(predictions[name])
        )
    return arrays, forward, parameters


@pytest.fixture
def prepared_case(joint, model, synthetic_context, monkeypatch, tmp_path):
    """Keep real joint storage/operators while replacing the PR4 opacity source."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "prepare.json").write_text('{"fixture": "Synthetic spectrum source"}')
    source_metadata = {
        "model_sha256": "fixture-model-hash",
        "artifacts": {"case_sha256": "fixture-source-array-hash"},
    }
    monkeypatch.setattr(
        joint,
        "_source_context",
        lambda *args: (source, source_metadata, {}, synthetic_context),
    )
    monkeypatch.setattr(joint.common, "load_case", lambda *args: (source_metadata, {}))
    monkeypatch.setattr(
        joint.common,
        "load_context",
        lambda *args: (source_metadata, {}, synthetic_context),
    )
    monkeypatch.setattr(
        joint, "_provenance", lambda *args: {"code_sha256": "fixture-revision"}
    )
    directory = tmp_path / "joint_case"
    arguments = joint._parser().parse_args(
        ["prepare", "--output-dir", str(directory), "--seed", "17"]
    )
    joint.prepare(arguments)
    return directory, arguments, source


def _run_arguments(joint, directory, method="nuts", run_id="repeat_a", seed=23):
    return joint._parser().parse_args(
        [
            "run",
            "--output-dir",
            str(directory),
            "--method",
            method,
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--num-chains",
            "2",
            "--num-warmup",
            "8",
            "--num-samples",
            "12",
        ]
    )


def _fake_chains(model, seed=23):
    rng = np.random.default_rng(seed)
    arrays = {
        f"samples__{name}": rng.uniform(prior["low"], prior["high"], (2, 12))
        for name, prior in model.JOINT_PRIORS.items()
    }
    arrays.update(
        extra__diverging=np.zeros((2, 12), dtype=bool),
        extra__num_steps=np.full((2, 12), 3, dtype=np.int32),
        extra__accept_prob=np.full((2, 12), 0.8),
    )
    return {"quality_passed": False, "fixture": True}, arrays


def _fake_nested(joint, model):
    adapter = importlib.import_module("_compare_samplers_jaxns")
    unit = np.tile(np.asarray([0.2, 0.5, 0.8])[:, None], (1, len(model.JOINT_PRIORS)))
    points = [joint.unit_to_physical(point, model.JOINT_PRIORS) for point in unit]
    raw = SimpleNamespace(
        samples={
            name: np.asarray([point[name] for point in points])
            for name in model.JOINT_PRIORS
        },
        log_dp_mean=np.log([0.1, 0.2, 0.7]),
        log_L_samples=np.asarray([-2.0, -1.0, -2.0]),
        U_samples=unit,
        log_X_mean=np.asarray([-1.0, -2.0, -3.0]),
        num_live_points_per_sample=np.asarray([8, 8, 7]),
        num_likelihood_evaluations_per_sample=np.asarray([1, 2, 3]),
        total_num_samples=3,
        log_Z_mean=-2.0,
        log_Z_uncert=0.1,
        ESS=1.25,
        termination_reason=1,
        total_num_likelihood_evaluations=6,
        total_phantom_samples=0,
    )
    report, arrays = adapter._summarize_results(raw)
    report["parameter_order"] = list(model.JOINT_PRIORS)
    return report, arrays


def test_import_has_no_sampler_or_process_setting_side_effects(monkeypatch):
    import jax

    original_import = builtins.__import__

    def without_samplers(name, *args, **kwargs):
        if name.split(".")[0] in {"numpyro", "jaxns", "tensorflow_probability"}:
            raise ImportError("Optional sampler intentionally unavailable")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must preserve process settings")

    with monkeypatch.context() as isolated:
        isolated.syspath_prepend(str(SCRIPT.parent))
        isolated.setattr(jax.config, "update", forbidden)
        isolated.setattr(os, "chdir", forbidden)
        isolated.setattr(builtins, "__import__", without_samplers)
        runpy.run_path(str(SCRIPT), run_name="joint_import_contract")


@pytest.mark.parametrize("subcommand", [None, "prepare", "run", "summarize"])
def test_cli_help_does_not_load_databases_or_samplers(subcommand):
    arguments = [] if subcommand is None else [subcommand]
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not Path(".database").exists()


def test_default_synthetic_co_case_uses_real_opacity_rt_and_saved_responses(
    joint, tmp_path
):
    """No source stubs: save/reload the small genuine PreMODIT and CIA model."""
    import jax
    import jax.numpy as jnp

    directory = tmp_path / "actual_joint_case"
    arguments = joint._parser().parse_args(
        ["prepare", "--output-dir", str(directory), "--seed", "17"]
    )
    joint.prepare(arguments)
    metadata, arrays, context = joint.load_context(directory)
    forward = joint.joint.make_forward(context, arrays)
    assert joint._check_fixed_points(forward, metadata, arrays)["passed"]
    assert (directory / "co_source" / "premodit.npz").is_file()
    assert len(arrays["observed_a"]) == 64
    assert len(arrays["observed_b"]) == 32
    assert not Path(".database").exists()

    def likelihood(temperature):
        return joint.joint.log_likelihood(
            forward, arrays, {**metadata["truth"], "T0": temperature}
        )

    temperature = jnp.asarray(metadata["truth"]["T0"])
    derivative = jax.grad(likelihood)(temperature)
    assert np.isfinite(derivative)
    assert abs(float(derivative)) > 1.0e-6
    for step in (0.01, 0.001):
        central = (likelihood(temperature + step) - likelihood(temperature - step)) / (
            2.0 * step
        )
        np.testing.assert_allclose(derivative, central, rtol=1.0e-5, atol=1.0e-8)


def test_two_seeded_observations_reload_and_refuse_overwrite(joint, prepared_case):
    directory, arguments, _ = prepared_case
    metadata, arrays, _ = joint.load_context(directory)
    _, restored = joint.load_case(directory)
    assert metadata["fixed_parameters"] == {"offset_a": 0.0}
    assert metadata["observation_seed"] == 17
    assert metadata["local_identifiability"]["passed"]
    rng = np.random.default_rng(17)
    for name in ("a", "b"):
        error = arrays[f"error_{name}"]
        expected_noise = rng.normal(0.0, error, error.shape)
        np.testing.assert_allclose(
            arrays[f"observed_{name}"] - arrays[f"mean_{name}"],
            expected_noise,
            atol=1.0e-10,
        )
        assert arrays[f"probe_mean_{name}"].shape == (3, len(error))
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, restored[name])
        assert value.dtype == restored[name].dtype
    before = (directory / "case.npz").read_bytes()
    with pytest.raises(FileExistsError):
        joint.prepare(arguments)
    assert (directory / "case.npz").read_bytes() == before


@pytest.mark.parametrize(
    "changed", ["case", "source", "instruments", "anchor", "prior"]
)
def test_changed_joint_data_source_or_definition_is_rejected(
    joint, prepared_case, changed
):
    directory, _, source = prepared_case
    path = directory / "prepare.json"
    metadata = json.loads(path.read_text())
    if changed in ("case", "source"):
        target = (
            directory / "case.npz" if changed == "case" else source / "prepare.json"
        )
        target.write_bytes(target.read_bytes() + b" changed")
    else:
        if changed == "instruments":
            metadata["instruments"]["b"]["resolution"] += 1.0
        elif changed == "anchor":
            metadata["fixed_parameters"]["offset_a"] = 1.0
        else:
            metadata["priors"]["scale_a"]["high"] += 1.0
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest"):
        joint.load_case(directory)


def test_rehashed_units_cannot_change_the_declared_flux_measure(joint, prepared_case):
    directory, _, _ = prepared_case
    path = directory / "prepare.json"
    metadata = json.loads(path.read_text())
    assert "sigmain" not in metadata["units"]
    metadata["units"]["flux"] = "Flux density per wavelength instead of per wavenumber"
    metadata["model_sha256"] = joint.common._digest(joint._definition(metadata))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="units"):
        joint.load_case(directory)


def test_rehashed_unsafe_bins_are_rejected_before_sampling(
    joint, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    path = directory / "case.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["bins_a"][0, 0] -= 1.0
    joint.common._storage().write_npz(path, **arrays)
    metadata_path = directory / "prepare.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["case_sha256"] = joint.common._storage().sha256(path)
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        joint,
        "_run_nuts",
        lambda *args: pytest.fail("Must validate bins before sampling"),
    )
    with pytest.raises(ValueError, match="geometry"):
        joint.run(_run_arguments(joint, directory))
    result = json.loads((directory / "runs/repeat_a/nuts/result.json").read_text())
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "input_validation"


def test_interrupted_sampler_is_saved_and_cannot_be_overwritten(
    joint, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case

    def interrupt(*args):
        raise RuntimeError("Synthetic sampler interruption")

    monkeypatch.setattr(joint, "_run_nuts", interrupt)
    arguments = _run_arguments(joint, directory)
    with pytest.raises(RuntimeError, match="interruption"):
        joint.run(arguments)
    path = directory / "runs/repeat_a/nuts/result.json"
    result = json.loads(path.read_text())
    assert result["failure"]["stage"] == "sampling"
    assert result["fixed_points"]["passed"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        joint.run(arguments)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="not completed"):
        joint.load_run(directory, "nuts", "repeat_a")


@pytest.mark.parametrize("method", ["nuts", "jaxns"])
def test_raw_samples_and_summary_keep_sampler_quality_separate(
    joint, model, prepared_case, monkeypatch, method
):
    directory, _, _ = prepared_case
    original = _fake_chains(model) if method == "nuts" else _fake_nested(joint, model)
    monkeypatch.setattr(joint, f"_run_{method}", lambda *args: original)
    joint.run(_run_arguments(joint, directory, method))
    report, restored = joint.load_run(directory, method, "repeat_a")
    for name, value in original[1].items():
        np.testing.assert_array_equal(restored[name], value)
        assert restored[name].dtype == value.dtype
    assert report["status"] == "completed"
    args = joint._parser().parse_args(
        [
            "summarize",
            "--output-dir",
            str(directory),
            "--method",
            method,
            "--run-id",
            "repeat_a",
        ]
    )
    joint.summarize(args)
    summary = joint.load_summary(directory, method, "repeat_a")
    assert not summary["quality_passed"]
    assert summary["parameters"]["offset_b"]["role"] == "instrument"
    assert summary["parameters"]["T0"]["role"] == "shared_atmosphere"
    assert summary["parameters"]["offset_b"]["truth"] == 250.0
    if method == "nuts":
        assert "ess_bulk" in summary["diagnostics"]["per_parameter"]["T0"]
        assert restored["samples__T0"].shape == (2, 12)
    else:
        assert not summary["diagnostics"]["converged"]
        assert summary["diagnostics"]["weighted_ess"] == pytest.approx(1.0 / 0.54)
        assert summary["diagnostics"]["weighted_ess"] != 3
    # Reading a summary recomputes intervals from its selected original arrays.
    path = directory / "runs" / "repeat_a" / method / "summary.json"
    forged = json.loads(path.read_text())
    forged["quality_passed"] = True
    forged["parameters"]["T0"]["mean"] = -1.0
    path.write_text(json.dumps(forged))
    recomputed = joint.load_summary(directory, method, "repeat_a")
    assert not recomputed["quality_passed"]
    assert recomputed["parameters"]["T0"]["mean"] > 1000.0
    json.dumps(recomputed, allow_nan=False)


def test_independent_seeds_remain_distinct_and_sample_tampering_is_rejected(
    joint, model, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    monkeypatch.setattr(
        joint,
        "_run_nuts",
        lambda forward, arrays, metadata, args: _fake_chains(model, args.seed),
    )
    saved = {}
    for run_id, seed in (("repeat_a", 23), ("repeat_b", 24)):
        joint.run(_run_arguments(joint, directory, run_id=run_id, seed=seed))
        report, saved[run_id] = joint.load_run(directory, "nuts", run_id)
        assert report["seed"] == seed
    assert not np.array_equal(
        saved["repeat_a"]["samples__T0"], saved["repeat_b"]["samples__T0"]
    )
    archive = directory / "runs/repeat_a/nuts/samples.npz"
    archive.write_bytes(archive.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="digest"):
        joint.load_run(directory, "nuts", "repeat_a")
    joint.load_run(directory, "nuts", "repeat_b")


def test_same_instrument_limit_and_independent_lsf(model, synthetic_context):
    instruments = copy.deepcopy(model.INSTRUMENTS)
    instruments["b"] = copy.deepcopy(instruments["a"])
    arrays = model.make_geometry(synthetic_context, instruments=instruments)
    parameters = {**model.mock_truth(), "offset_b": 0.0}
    prediction = model.make_forward(synthetic_context, arrays, instruments=instruments)(
        parameters
    )
    np.testing.assert_array_equal(prediction["a"], prediction["b"])

    instruments["b"]["resolution"] = 20000.0
    changed = model.make_forward(synthetic_context, arrays, instruments=instruments)(
        parameters
    )
    np.testing.assert_array_equal(prediction["a"], changed["a"])
    assert np.max(np.abs(changed["a"] - changed["b"])) > 1.0


@pytest.mark.parametrize("field", ["sample_grid", "bins_a", "error_b"])
def test_changed_geometry_or_incomplete_response_coverage_is_rejected(
    model, synthetic_context, field
):
    arrays = model.make_geometry(synthetic_context)
    if field == "sample_grid":
        arrays[field][0] = synthetic_context["nu_grid"][0]
        message = "coverage"
    else:
        arrays[field].flat[0] += 0.01
        message = field
    with pytest.raises(ValueError, match=message):
        model.make_forward(synthetic_context, arrays)


def test_joint_observation_ad_matches_resolved_central_differences(
    model, synthetic_case
):
    import jax
    import jax.numpy as jnp

    arrays, forward, parameters = synthetic_case

    def predictions(coordinates):
        point = {
            **parameters,
            "T0": 1250.0 + 100.0 * coordinates[0],
            "RV": 39.0 + coordinates[1],
        }
        result = forward(point)
        return jnp.concatenate(
            [result[name] / arrays[f"error_{name}"] for name in ("a", "b")]
        )

    origin = jnp.zeros(2)
    direction = jnp.asarray([0.4, 0.7])
    derivative = jax.jvp(predictions, (origin,), (direction,))[1]
    assert np.all(np.isfinite(derivative))
    assert np.linalg.norm(derivative) > 0.01
    for step in (1.0e-3, 1.0e-4):
        central = (predictions(step * direction) - predictions(-step * direction)) / (
            2.0 * step
        )
        np.testing.assert_allclose(derivative, central, rtol=1.0e-6, atol=1.0e-8)


def test_each_instrument_has_a_normalized_gaussian_likelihood(model, synthetic_case):
    arrays, forward, parameters = synthetic_case
    predictions = forward(parameters)
    expected = 0.0
    for name in ("a", "b"):
        sigma = arrays[f"error_{name}"] * parameters[f"scale_{name}"]
        residual = arrays[f"observed_{name}"] - np.asarray(predictions[name])
        expected -= 0.5 * np.sum(
            (residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)
        )
    np.testing.assert_allclose(
        model.log_likelihood(forward, arrays, parameters), expected, rtol=1.0e-13
    )
    # At zero residual, the normalization penalizes a larger uncertainty scale.
    exact = {**arrays, **{f"observed_{name}": predictions[name] for name in ("a", "b")}}
    widened = {**parameters, "scale_b": 2.0 * parameters["scale_b"]}
    difference = model.log_likelihood(forward, exact, widened) - model.log_likelihood(
        forward, exact, parameters
    )
    np.testing.assert_allclose(
        difference, -len(predictions["b"]) * np.log(2.0), rtol=1.0e-13
    )
    for name in ("scale_a", "scale_b"):
        assert np.isneginf(
            model.log_likelihood(forward, arrays, {**parameters, name: 0.0})
        )


def test_anchored_offset_and_error_scales_are_locally_distinct(model, synthetic_case):
    """A local nuisance-parameter rank check is not a posterior recovery claim."""
    import jax
    import jax.numpy as jnp

    arrays, forward, parameters = synthetic_case
    original = importlib.import_module("_co_retrieval")
    assert set(model.JOINT_PRIORS) == (set(original.PRIOR_SPECS) - {"sigmain"}) | {
        "offset_b",
        "scale_a",
        "scale_b",
    }
    for name in set(original.PRIOR_SPECS) - {"sigmain"}:
        assert model.JOINT_PRIORS[name] == original.PRIOR_SPECS[name]

    def mean_and_error(nuisance):
        point = {
            **parameters,
            "offset_b": nuisance[0],
            "scale_a": nuisance[1],
            "scale_b": nuisance[2],
        }
        means = forward(point)
        errors = model.error_scales(arrays, point)
        return jnp.concatenate([means["a"], means["b"], errors["a"], errors["b"]])

    derivative = np.asarray(jax.jacfwd(mean_and_error)(jnp.asarray([175.0, 0.8, 1.3])))
    normalized = derivative / np.linalg.norm(derivative, axis=0)
    assert np.linalg.matrix_rank(normalized, tol=1.0e-10) == 3
    np.testing.assert_allclose(normalized.T @ normalized, np.eye(3), atol=1.0e-14)
    changed = forward({**parameters, "offset_b": parameters["offset_b"] + 10.0})
    original_means = forward(parameters)
    np.testing.assert_array_equal(changed["a"], original_means["a"])
    np.testing.assert_allclose(changed["b"] - original_means["b"], 10.0)


def test_numpyro_joint_entry_matches_the_same_physical_density(model, synthetic_case):
    pytest.importorskip("numpyro")
    from numpyro.infer.util import log_density

    arrays, forward, parameters = synthetic_case
    observation = np.concatenate([arrays["observed_a"], arrays["observed_b"]])
    joint_density, trace = log_density(
        model.make_numpyro_model(forward, arrays, model.JOINT_PRIORS),
        (observation,),
        {},
        parameters,
    )
    co = importlib.import_module("_co_retrieval")
    likelihood = trace["spectrum"]["fn"].log_prob(observation).sum()
    np.testing.assert_allclose(
        likelihood, model.log_likelihood(forward, arrays, parameters), rtol=1.0e-13
    )
    np.testing.assert_allclose(
        joint_density - likelihood,
        co.normalized_log_prior(parameters, model.JOINT_PRIORS),
        rtol=1.0e-13,
    )
    assert set(trace) == set(model.JOINT_PRIORS) | {"spectrum"}
    assert trace["spectrum"]["is_observed"]


def test_jaxns_joint_entry_matches_the_same_physical_density(model, synthetic_case):
    try:
        version = importlib.metadata.version("jaxns")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("JAXNS is optional; exercised in the separate sampler environment.")
    adapter = importlib.import_module("_compare_samplers_jaxns")
    if version != adapter.SUPPORTED_JAXNS_VERSION:
        pytest.skip("This contract uses the explicitly supported JAXNS API.")

    arrays, forward, parameters = synthetic_case
    co = importlib.import_module("_co_retrieval")
    unit = co.physical_to_unit(parameters, model.JOINT_PRIORS)
    nested_model = adapter.build_jaxns_model(
        lambda point: model.log_likelihood(forward, arrays, point), model.JOINT_PRIORS
    )
    transformed = nested_model.transform(unit)
    for name, expected in parameters.items():
        np.testing.assert_allclose(transformed[name], expected, rtol=1.0e-13)
    np.testing.assert_allclose(
        nested_model.forward(unit),
        model.log_likelihood(forward, arrays, parameters),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        nested_model.log_prob_prior(unit),
        co.normalized_log_prior(parameters, model.JOINT_PRIORS),
        rtol=1.0e-13,
    )
