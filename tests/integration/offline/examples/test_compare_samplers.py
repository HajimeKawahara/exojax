"""Offline contracts for the shared CO NUTS and nested-sampling example."""

import builtins
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


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "examples" / "compare_samplers.py"


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("compare_samplers")


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("_co_retrieval")


@pytest.fixture
def physical_parameters():
    return {
        "logg": 4.4,
        "RV": 39.0,
        "MMR": 0.008,
        "T0": 1200.0,
        "alpha": 0.11,
        "vsini": 11.0,
        "sigmain": 550.0,
    }


@pytest.fixture
def tiny_real_context():
    """Three synthetic CO lines run through the real opacity, RT, and operators."""
    import jax.numpy as jnp

    from exojax.opacity import OpaCIA, OpaDirect
    from exojax.postproc.specop import SopInstProfile, SopRotation
    from exojax.rt import ArtEmisPure
    from exojax.utils.instfunc import resolution_to_gaussian_std

    nu_grid = np.geomspace(4330.0, 4350.0, 256)
    partition_temperatures = jnp.asarray([200.0, 500.0, 1000.0, 2000.0, 3000.0])
    partition_values = jnp.asarray([70.0, 180.0, 380.0, 850.0, 1400.0])
    mdb = SimpleNamespace(
        dbtype="exomol",
        molmass=28.0,
        nu_lines=jnp.asarray([4335.0, 4340.0, 4345.0]),
        elower=jnp.asarray([200.0, 500.0, 900.0]),
        logsij0=jnp.log(jnp.asarray([2.0e-22, 3.0e-22, 1.0e-22])),
        A=jnp.asarray([1.0, 2.0, 1.5]),
        n_Texp=jnp.asarray([0.5, 0.55, 0.6]),
        alpha_ref=jnp.asarray([0.05, 0.06, 0.07]),
        qr_interp=lambda temperature, reference: (
            jnp.interp(temperature, partition_temperatures, partition_values)
            / jnp.interp(reference, partition_temperatures, partition_values)
        ),
    )
    cdb = SimpleNamespace(
        nucia=jnp.asarray([4320.0, 4360.0]),
        tcia=jnp.asarray([200.0, 3000.0]),
        logac=jnp.full((2, 2), -46.0),
    )
    art = ArtEmisPure(pressure_top=0.1, pressure_btm=10.0, nlayer=3, nu_grid=nu_grid)
    art.change_temperature_range(400.0, 2000.0)
    return {
        "art": art,
        "opa": OpaDirect(mdb, nu_grid),
        "opacia": OpaCIA(cdb, nu_grid),
        "sop_rot": SopRotation(nu_grid, vsini_max=20.0),
        "sop_inst": SopInstProfile(nu_grid, vrmax=50.0),
        "beta_inst": resolution_to_gaussian_std(70000.0),
        "nu_obs": jnp.asarray(nu_grid[64:192:16]),
        "molmass": mdb.molmass,
        "vmrH2": 0.855,
        "mmw": 2.33,
    }


def _spectrum(fspec, parameters):
    return fspec(
        parameters["T0"],
        parameters["alpha"],
        parameters["MMR"],
        10 ** parameters["logg"],
        parameters["RV"],
        parameters["vsini"],
    )


@pytest.fixture
def prepared_case(comparison, monkeypatch, tmp_path):
    """Exercise real CLI storage with a cheap synthetic spectrum and local inputs."""
    import jax.numpy as jnp

    directory = tmp_path / "shared_case"
    cia_path = tmp_path / "synthetic.cia"
    cia_path.write_bytes(b"Synthetic local continuum input")
    nu_grid = np.geomspace(1.0e8 / 23000.0, 1.0e8 / 22920.0, 16)
    context = {
        "opa": object(),
        "nu_grid": nu_grid,
        "nu_obs": nu_grid[::2][:-2],
        "cia_path": cia_path,
        "source_info": {"fixture": "Synthetic linear spectrum; no database download"},
    }

    def forward(T0, alpha, MMR, gravity, RV, vsini):
        return jnp.asarray(
            [
                T0,
                alpha * 1.0e4,
                MMR * 1.0e5,
                jnp.log10(gravity) * 300,
                RV * 30,
                vsini * 100,
            ]
        )

    def save_opacity(opacity, path):
        np.savez(path, fixture_lines=np.asarray([1.0, 2.0, 3.0]))
        path.with_name("premodit_metadata.json").write_text(
            json.dumps({"fixture": "Synthetic archive used only for hash validation"})
        )

    monkeypatch.setattr(comparison, "_prepare_context", lambda *args: context)
    monkeypatch.setattr(comparison, "build_context", lambda *args: context)
    monkeypatch.setattr(comparison, "_save_opacity", save_opacity)
    monkeypatch.setattr(comparison, "make_forward", lambda _: forward)
    monkeypatch.setattr(
        comparison,
        "_provenance",
        lambda *args: {"code_sha256": "fixture-model-revision", "dependencies": {}},
    )
    args = comparison._parser().parse_args(
        [
            "prepare",
            "--output-dir",
            str(directory),
            "--cia-path",
            str(cia_path),
            "--number-of-wavenumbers",
            "16",
            "--number-of-layers",
            "3",
            "--observation-stride",
            "2",
            "--observation-trim",
            "2",
            "--seed",
            "17",
        ]
    )
    comparison.prepare(args)
    return directory, args, context


def _run_arguments(comparison, directory, method="nuts", run_id="repeat_a", seed=11):
    return comparison._parser().parse_args(
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


def _fake_chain_result(comparison, seed=0):
    rng = np.random.default_rng(seed)
    samples = {}
    for name, prior in comparison.PRIOR_SPECS.items():
        if prior["distribution"] == "uniform":
            values = rng.uniform(prior["low"], prior["high"], size=(2, 12))
        else:
            values = rng.exponential(1.0 / prior["rate"], size=(2, 12))
        samples[f"samples__{name}"] = values
    samples["extra__diverging"] = np.zeros((2, 12), dtype=bool)
    samples["extra__num_steps"] = np.full((2, 12), 3, dtype=np.int32)
    samples["extra__accept_prob"] = np.full((2, 12), 0.8)
    return {"quality_passed": False, "fixture": True}, samples


def _fake_nested_result(comparison):
    adapter = importlib.import_module("_compare_samplers_jaxns")
    parameters = [
        comparison.unit_to_physical(np.full(len(comparison.PRIOR_SPECS), fraction))
        for fraction in (0.2, 0.5, 0.8)
    ]
    raw = SimpleNamespace(
        samples={
            name: np.asarray([point[name] for point in parameters])
            for name in comparison.PRIOR_SPECS
        },
        log_dp_mean=np.log([0.1, 0.2, 0.7]),
        log_L_samples=np.asarray([-2.0, -1.0, -2.0]),
        U_samples=np.asarray([np.full(7, fraction) for fraction in (0.2, 0.5, 0.8)]),
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
    return adapter._summarize_results(raw)


def _replace_saved_arrays(comparison, directory, method, transform):
    """Keep byte hashes valid to test the semantic archive contract separately."""
    target = directory / "runs" / "repeat_a" / method
    with np.load(target / "samples.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    transform(arrays)
    comparison._storage().write_npz(target / "samples.npz", **arrays)
    report_path = target / "result.json"
    report = json.loads(report_path.read_text())
    report["samples"]["sha256"] = comparison._storage().sha256(target / "samples.npz")
    report["samples"]["arrays"] = {
        name: {"shape": list(value.shape), "dtype": value.dtype.str}
        for name, value in arrays.items()
    }
    report_path.write_text(json.dumps(report))


def test_import_does_not_change_process_settings_or_require_samplers(monkeypatch):
    import jax

    original_import = builtins.__import__

    def import_without_samplers(name, *args, **kwargs):
        if name.split(".")[0] in {"numpyro", "jaxns", "tensorflow_probability"}:
            raise ImportError("Optional sampler intentionally unavailable")
        return original_import(name, *args, **kwargs)

    def forbidden_change(*args, **kwargs):
        raise AssertionError("Import must not change process settings")

    with monkeypatch.context() as isolated:
        isolated.syspath_prepend(str(SCRIPT.parent))
        isolated.setattr(jax.config, "update", forbidden_change)
        isolated.setattr(os, "chdir", forbidden_change)
        isolated.setattr(builtins, "__import__", import_without_samplers)
        runpy.run_path(str(SCRIPT), run_name="comparison_import_contract")


@pytest.mark.parametrize("subcommand", [None, "prepare", "run", "summarize"])
def test_cli_help_needs_no_database_or_sampler(subcommand):
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


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seed", "-1"],
        ["--seed", str(2**32)],
        ["--run-id", "../outside"],
        ["--dlogz", "nan"],
        ["--num-chains", "0"],
    ],
)
def test_invalid_run_cli_arguments_are_rejected(comparison, arguments):
    with pytest.raises(SystemExit):
        comparison._parser().parse_args(
            ["run", "--method", "nuts", "--run-id", "selected", *arguments]
        )


def test_run_seed_accepts_the_full_uint32_range(comparison):
    for seed in (0, 2**32 - 1):
        args = comparison._parser().parse_args(
            ["run", "--method", "nuts", "--run-id", "selected", "--seed", str(seed)]
        )
        assert args.seed == seed


def test_preparation_reuses_one_seeded_observation_and_preserves_dtypes(
    comparison, prepared_case
):
    directory, args, _ = prepared_case
    metadata, first = comparison.load_case(directory)
    _, second = comparison.load_case(directory)
    expected_noise = np.random.default_rng(17).normal(0.0, 500.0, 6)
    np.testing.assert_allclose(
        first["observed_flux"] - first["mean_flux"], expected_noise
    )
    assert metadata["observation_seed"] == 17
    assert metadata["priors"] == comparison.PRIOR_SPECS
    assert metadata["status"] == "completed"
    for name, array in first.items():
        np.testing.assert_array_equal(second[name], array)
        assert second[name].dtype == array.dtype == np.float64
    assert first["probe_flux"].shape == (3, 6)
    assert first["probe_log_likelihood"].shape == first["probe_log_prior"].shape == (3,)
    assert first["noise_sigma"] == 500.0
    before = (directory / "case.npz").read_bytes()
    with pytest.raises(FileExistsError):
        comparison.prepare(args)
    assert (directory / "case.npz").read_bytes() == before


@pytest.mark.parametrize(
    "target", ["case.npz", "premodit.npz", "premodit_metadata.json", "cia"]
)
def test_loading_rejects_changed_observation_or_physical_input(
    comparison, prepared_case, target
):
    directory, _, context = prepared_case
    path = context["cia_path"] if target == "cia" else directory / target
    path.write_bytes(path.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="digest"):
        comparison.load_case(directory)


@pytest.mark.parametrize("field", ["config", "priors", "units", "model_sha256"])
def test_loading_rejects_changed_model_or_prior_metadata(
    comparison, prepared_case, field
):
    directory, _, _ = prepared_case
    path = directory / "prepare.json"
    metadata = json.loads(path.read_text())
    if field == "config":
        metadata[field]["number_of_layers"] += 1
    elif field == "priors":
        metadata[field]["T0"]["low"] -= 1
    elif field == "units":
        metadata[field]["flux"] = "Different flux units"
    else:
        metadata[field] = "changed-model"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest"):
        comparison.load_case(directory)


def test_failed_sampler_is_recorded_and_named_run_is_not_overwritten(
    comparison, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    args = _run_arguments(comparison, directory)

    def fail(*arguments):
        raise RuntimeError("Synthetic sampler interruption")

    monkeypatch.setattr(comparison, "_run_nuts", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        comparison.run(args)
    result_path = directory / "runs" / "repeat_a" / "nuts" / "result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "sampling"
    assert result["failure"]["type"] == "RuntimeError"
    assert result["fixed_points"]["passed"]
    assert not result_path.with_name("samples.npz").exists()
    before = result_path.read_bytes()
    with pytest.raises(FileExistsError):
        comparison.run(args)
    assert result_path.read_bytes() == before
    with pytest.raises(ValueError, match="not completed"):
        comparison.load_run(directory, "nuts", "repeat_a")


def test_changed_forward_is_rejected_before_sampling_and_failure_is_saved(
    comparison, prepared_case, monkeypatch
):
    import jax.numpy as jnp

    directory, _, _ = prepared_case
    monkeypatch.setattr(
        comparison, "make_forward", lambda _: lambda *parameters: jnp.zeros(6)
    )
    monkeypatch.setattr(
        comparison,
        "_run_nuts",
        lambda *args: pytest.fail("Must reject before sampling"),
    )
    with pytest.raises(ValueError, match="fixed points"):
        comparison.run(_run_arguments(comparison, directory))
    result_path = directory / "runs" / "repeat_a" / "nuts" / "result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "input_validation"


def test_saved_chains_round_trip_and_independent_seeds_are_recorded(
    comparison, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    original = {}

    def sample(forward, observation, metadata, args):
        report, arrays = _fake_chain_result(comparison, args.seed)
        original[args.run_id] = arrays
        return report, arrays

    monkeypatch.setattr(comparison, "_run_nuts", sample)
    for run_id, seed in (("repeat_a", 11), ("repeat_b", 12)):
        comparison.run(_run_arguments(comparison, directory, run_id=run_id, seed=seed))
        report, arrays = comparison.load_run(directory, "nuts", run_id)
        assert report["seed"] == seed
        assert report["settings"]["num_chains"] == 2
        assert report["status"] == "completed"
        assert not report["nuts"]["quality_passed"]
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, original[run_id][name])
            assert array.dtype == original[run_id][name].dtype
            assert array.shape == (2, 12)
    assert not np.array_equal(
        original["repeat_a"]["samples__T0"], original["repeat_b"]["samples__T0"]
    )


@pytest.mark.parametrize(
    "changed", ["sample_bytes", "sample_dtype", "case_sha256", "model_sha256"]
)
def test_saved_run_rejects_archive_or_model_contract_mismatch(
    comparison, prepared_case, monkeypatch, changed
):
    directory, _, _ = prepared_case
    monkeypatch.setattr(
        comparison, "_run_nuts", lambda *args: _fake_chain_result(comparison)
    )
    comparison.run(_run_arguments(comparison, directory))
    target = directory / "runs" / "repeat_a" / "nuts"
    path = target / "result.json"
    report = json.loads(path.read_text())
    if changed == "sample_bytes":
        archive = target / "samples.npz"
        archive.write_bytes(archive.read_bytes() + b" changed")
    elif changed == "sample_dtype":
        report["samples"]["arrays"]["samples__T0"]["dtype"] = "<f4"
    else:
        report[changed] = "wrong-prepared-model"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs"):
        comparison.load_run(directory, "nuts", "repeat_a")


def test_rehashed_flattened_chain_cannot_bypass_the_chain_contract(
    comparison, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    monkeypatch.setattr(
        comparison, "_run_nuts", lambda *args: _fake_chain_result(comparison)
    )
    comparison.run(_run_arguments(comparison, directory))

    def flatten(arrays):
        arrays["samples__T0"] = arrays["samples__T0"].reshape(-1)

    _replace_saved_arrays(comparison, directory, "nuts", flatten)
    with pytest.raises(ValueError, match="chain|shape|dimensions"):
        comparison.load_run(directory, "nuts", "repeat_a")


@pytest.mark.parametrize(
    "changed", ["normalized_weights", "nonfinite_weights", "unit_samples"]
)
def test_rehashed_nested_archive_cannot_change_original_weight_or_prior_contract(
    comparison, prepared_case, monkeypatch, changed
):
    directory, _, _ = prepared_case
    monkeypatch.setattr(
        comparison, "_run_jaxns", lambda *args: _fake_nested_result(comparison)
    )
    comparison.run(_run_arguments(comparison, directory, method="jaxns"))

    def alter(arrays):
        if changed == "normalized_weights":
            arrays["weights"] = np.asarray([0.8, 0.1, 0.1])
        elif changed == "nonfinite_weights":
            arrays["weights"][0] = np.nan
        else:
            arrays["unit_samples"][0, 0] = 0.9

    _replace_saved_arrays(comparison, directory, "jaxns", alter)
    with pytest.raises(ValueError):
        comparison.load_run(directory, "jaxns", "repeat_a")


def test_nonfinite_chain_cannot_produce_a_nan_successful_comparison(
    comparison, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    monkeypatch.setattr(
        comparison, "_run_nuts", lambda *args: _fake_chain_result(comparison)
    )
    monkeypatch.setattr(
        comparison, "_run_jaxns", lambda *args: _fake_nested_result(comparison)
    )
    for method in ("nuts", "jaxns"):
        comparison.run(_run_arguments(comparison, directory, method=method))

    def nonfinite(arrays):
        arrays["samples__T0"][0, 0] = np.nan

    _replace_saved_arrays(comparison, directory, "nuts", nonfinite)
    args = comparison._parser().parse_args(
        ["summarize", "--output-dir", str(directory), "--run-id", "repeat_a"]
    )
    failed = False
    try:
        comparison.summarize(args)
    except ValueError:
        failed = True
    report = json.loads(
        (directory / "runs" / "repeat_a" / "comparison.json").read_text()
    )
    json.dumps(report, allow_nan=False)
    if failed:
        assert report["status"] == "failed"
    else:
        assert not report["nuts"]["nuts"]["quality_passed"]
        assert report["posterior_means"]["T0"]["nuts"] is None


def test_sampler_comparison_keeps_chain_and_weight_diagnostics_distinct(
    comparison, prepared_case, monkeypatch
):
    directory, _, _ = prepared_case
    observations = []

    def chains(forward, observation, metadata, args):
        observations.append(np.asarray(observation).copy())
        return _fake_chain_result(comparison)

    def nested(forward, observation, metadata, args):
        observations.append(np.asarray(observation).copy())
        return _fake_nested_result(comparison)

    monkeypatch.setattr(comparison, "_run_nuts", chains)
    monkeypatch.setattr(comparison, "_run_jaxns", nested)
    for method in ("nuts", "jaxns"):
        comparison.run(_run_arguments(comparison, directory, method))
    np.testing.assert_array_equal(observations[0], observations[1])
    args = comparison._parser().parse_args(
        ["summarize", "--output-dir", str(directory), "--run-id", "repeat_a"]
    )
    comparison.summarize(args)
    target = directory / "runs" / "repeat_a"
    report = json.loads((target / "comparison.json").read_text())
    assert report["status"] == "completed"
    assert not report["nuts"]["nuts"]["quality_passed"]
    assert not report["jaxns"]["nested"]["converged"]
    assert report["jaxns"]["nested"]["status"] == "unconverged"
    assert report["jaxns"]["nested"]["weighted_ess"] == pytest.approx(1 / 0.54)
    assert report["jaxns"]["nested"]["weighted_ess"] != 3
    assert "ess_bulk" in next(iter(report["nuts"]["nuts"]["per_parameter"].values()))
    assert "speedup" not in report
    assert report["nuts"]["case_sha256"] == report["jaxns"]["case_sha256"]
    assert report["nuts"]["model_sha256"] == report["jaxns"]["model_sha256"]

    nested_path = target / "jaxns" / "result.json"
    changed = json.loads(nested_path.read_text())
    changed["provenance"]["code_sha256"] = "different-shared-model-revision"
    nested_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="code differs"):
        comparison.summarize(args)
    assert json.loads((target / "comparison.json").read_text())["status"] == "failed"


def test_real_co_forward_uses_the_same_normalized_likelihood(
    model, tiny_real_context, physical_parameters
):
    """The synthetic fixture includes CIA, rotation, LSF, RV, and sampling."""
    import jax
    import jax.numpy as jnp

    fspec = model.make_forward(tiny_real_context)
    prediction = _spectrum(fspec, physical_parameters)
    assert prediction.shape == tiny_real_context["nu_obs"].shape
    assert prediction.dtype == jnp.float64
    assert np.all(np.isfinite(prediction))
    assert np.ptp(prediction) > 0.0
    observation = prediction + jnp.linspace(-250.0, 250.0, len(prediction))
    sigma = physical_parameters["sigmain"]
    expected = -0.5 * np.sum(
        np.square((observation - prediction) / sigma) + np.log(2.0 * np.pi * sigma**2)
    )
    actual = model.physical_log_likelihood(fspec, observation, physical_parameters)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13)
    derivative = jax.grad(
        lambda temperature: model.physical_log_likelihood(
            fspec, observation, {**physical_parameters, "T0": temperature}
        )
    )(jnp.asarray(physical_parameters["T0"]))
    assert np.isfinite(derivative)


def test_physical_prior_keeps_tutorial_normalization_and_support(
    model, physical_parameters
):
    widths = np.asarray([1.0, 10.0, 0.015, 500.0, 0.15, 10.0])
    expected = -np.log(widths).sum() + np.log(1.0e-3) - 0.55
    np.testing.assert_allclose(
        model.normalized_log_prior(physical_parameters), expected, rtol=1.0e-14
    )
    for changed in ({"T0": 999.0}, {"MMR": -0.001}, {"sigmain": -1.0}):
        assert np.isneginf(
            model.normalized_log_prior({**physical_parameters, **changed})
        )


def test_unit_cube_transform_induces_the_same_physical_prior(model):
    import jax
    import jax.numpy as jnp

    unit = jnp.linspace(0.15, 0.85, len(model.PRIOR_SPECS))
    names = list(model.PRIOR_SPECS)
    parameters = model.unit_to_physical(unit)
    np.testing.assert_allclose(model.physical_to_unit(parameters), unit, atol=1.0e-15)
    sigma_index = names.index("sigmain")
    np.testing.assert_allclose(
        parameters["sigmain"], -np.log1p(-unit[sigma_index]) / 1.0e-3
    )
    derivative = jax.jacfwd(
        lambda value: jnp.asarray(
            [model.unit_to_physical(value)[name] for name in names]
        )
    )(unit)
    sign, log_determinant = jnp.linalg.slogdet(derivative)
    assert sign == 1.0
    np.testing.assert_allclose(
        model.unit_log_jacobian(unit), log_determinant, atol=1.0e-12
    )
    np.testing.assert_allclose(
        model.normalized_log_prior(parameters) + log_determinant,
        0.0,
        atol=1.0e-12,
    )


def test_numpyro_entry_matches_physical_likelihood_and_normalized_prior(
    model, tiny_real_context, physical_parameters
):
    pytest.importorskip("numpyro")
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    fspec = model.make_forward(tiny_real_context)
    observation = _spectrum(fspec, physical_parameters) + jnp.arange(
        len(tiny_real_context["nu_obs"])
    )
    joint, trace = log_density(
        model.make_numpyro_model(fspec), (observation,), {}, physical_parameters
    )
    actual_likelihood = trace["spectrum"]["fn"].log_prob(observation).sum()
    np.testing.assert_allclose(
        actual_likelihood,
        model.physical_log_likelihood(fspec, observation, physical_parameters),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        joint - actual_likelihood,
        model.normalized_log_prior(physical_parameters),
        rtol=1.0e-13,
    )
    assert trace["spectrum"]["is_observed"]


def test_jaxns_entry_matches_the_same_real_co_fixed_point(
    model, tiny_real_context, physical_parameters
):
    """Optional backend parity uses the same real CO fixture as the NUTS entry."""
    try:
        version = importlib.metadata.version("jaxns")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("JAXNS is optional; checked in the separate JAXNS environment.")
    adapter = importlib.import_module("_compare_samplers_jaxns")
    if version != adapter.SUPPORTED_JAXNS_VERSION:
        pytest.skip("The parity test requires the explicitly supported JAXNS API.")

    import jax.numpy as jnp

    fspec = model.make_forward(tiny_real_context)
    observation = _spectrum(fspec, physical_parameters) + jnp.arange(
        len(tiny_real_context["nu_obs"])
    )

    def likelihood(parameters):
        return model.physical_log_likelihood(fspec, observation, parameters)

    nested_model = adapter.build_jaxns_model(likelihood, model.PRIOR_SPECS)
    unit = model.physical_to_unit(physical_parameters)
    transformed = nested_model.transform(unit)
    for name, value in physical_parameters.items():
        np.testing.assert_allclose(transformed[name], value, rtol=1.0e-13)
    np.testing.assert_allclose(
        nested_model.forward(unit), likelihood(physical_parameters), rtol=1.0e-13
    )
    np.testing.assert_allclose(
        nested_model.log_prob_prior(unit),
        model.normalized_log_prior(physical_parameters),
        rtol=1.0e-13,
    )


def test_actual_numpyro_potential_includes_all_unconstrained_jacobians(model):
    pytest.importorskip("numpyro")
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import initialize_model

    # Constant flux isolates probability-coordinate conventions from RT accuracy.
    def fspec(*arguments):
        return jnp.asarray([100.0, 120.0, 80.0])

    observation = jnp.asarray([200.0, 80.0, 70.0])
    info = initialize_model(
        jax.random.PRNGKey(0),
        model.make_numpyro_model(fspec),
        model_args=(observation,),
    )
    bounds = {
        "logg": (4.0, 5.0),
        "RV": (35.0, 45.0),
        "MMR": (0.0, 0.015),
        "T0": (1000.0, 1500.0),
        "alpha": (0.05, 0.2),
        "vsini": (5.0, 15.0),
    }
    unconstrained = {
        name: jnp.asarray(-0.8 + index * 0.25) for index, name in enumerate(bounds)
    }
    unconstrained["sigmain"] = jnp.asarray(np.log(470.0))

    def expected_potential(coordinates):
        physical = {}
        log_jacobian = jnp.asarray(0.0)
        for name, (lower, upper) in bounds.items():
            value = coordinates[name]
            physical[name] = lower + (upper - lower) * jax.nn.sigmoid(value)
            log_jacobian += (
                jnp.log(upper - lower)
                + jax.nn.log_sigmoid(value)
                + jax.nn.log_sigmoid(-value)
            )
        physical["sigmain"] = jnp.exp(coordinates["sigmain"])
        log_jacobian += coordinates["sigmain"]
        return (
            -model.physical_log_likelihood(fspec, observation, physical)
            - model.normalized_log_prior(physical)
            - log_jacobian
        )

    np.testing.assert_allclose(
        info.potential_fn(unconstrained),
        expected_potential(unconstrained),
        rtol=1.0e-13,
    )
    actual_gradient = jax.grad(info.potential_fn)(unconstrained)
    expected_gradient = jax.grad(expected_potential)(unconstrained)
    for name in unconstrained:
        np.testing.assert_allclose(
            actual_gradient[name], expected_gradient[name], atol=1.0e-12
        )
