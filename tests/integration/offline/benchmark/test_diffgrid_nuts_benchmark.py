"""Offline contracts for the manual DiffGrid NUTS benchmark."""

import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = ROOT / "tests" / "benchmark"


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARK_DIR))
    return importlib.import_module("diffgrid_nuts_storage")


@pytest.fixture
def benchmark(monkeypatch, storage):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    module = importlib.import_module("diffgrid_nuts_benchmark")
    module._load_scientific_runtime()
    return module


@pytest.fixture
def prepared(tmp_path, benchmark, storage):
    paths = benchmark._case_paths(tmp_path)
    storage.write_npz(paths["case"], observed_flux=np.array([1.0, 1.1]))
    for name in ("premodit", "premodit_metadata", "diffgrid", "diffgrid_metadata"):
        paths[name].write_bytes(name.encode())
    cia_path = tmp_path / "synthetic.cia"
    cia_path.write_bytes(b"synthetic CIA")
    metadata = {
        "schema_version": 1,
        "config": {},
        "truth": benchmark.TRUTH,
        "prior_bounds": benchmark.PRIOR_BOUNDS,
        "artifacts": {
            f"{name}_sha256": storage.sha256(path)
            for name, path in paths.items()
            if name != "prepare"
        },
        "inputs": {"cia_path": str(cia_path), "cia_sha256": storage.sha256(cia_path)},
    }
    storage.write_json(paths["prepare"], metadata)
    return paths, metadata, cia_path


def test_cli_help_and_import_do_not_require_numpyro_or_change_precision(tmp_path):
    script = """
import builtins, runpy, sys
from pathlib import Path
import jax
original_import = builtins.__import__
def no_numpyro(name, *args, **kwargs):
    if name == 'numpyro' or name.startswith('numpyro.'):
        raise ModuleNotFoundError('NumPyro deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_numpyro
jax.config.update('jax_enable_x64', False)
module = runpy.run_path(sys.argv[1], run_name='benchmark_contract')
assert Path(module['exojax'].__file__).resolve().is_relative_to(Path(sys.argv[1]).parents[2] / 'src')
assert not jax.config.jax_enable_x64
assert 'exojax.special.j0' not in sys.modules
assert 'exojax.special.faddeeva' not in sys.modules
for command in ([], ['prepare'], ['run'], ['summarize']):
    try:
        module['_parser']().parse_args([*command, '--help'])
    except SystemExit as error:
        assert error.code == 0
assert not jax.config.jax_enable_x64
assert 'exojax.special.j0' not in sys.modules
jax.config.update('jax_enable_x64', True)
module['_load_scientific_runtime']()
assert str(sys.modules['exojax.special.j0'].RP.dtype) == 'float64'
assert str(sys.modules['exojax.special.faddeeva'].an.dtype) == 'float64'
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(BENCHMARK_DIR)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(BENCHMARK_DIR / "diffgrid_nuts_benchmark.py"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--run-id" in result.stdout
    assert "--num-chains" in result.stdout


def test_legacy_cli_arguments_and_defaults(benchmark, tmp_path):
    parser = benchmark._parser()
    prepare = parser.parse_args(
        [
            "prepare",
            "--output-dir",
            str(tmp_path),
            "--mdb-path",
            "ch4",
            "--cia-path",
            "cia",
            "--number-of-observed-wavenumbers",
            "8",
            "--number-of-wavenumbers",
            "256",
            "--number-of-layers",
            "4",
            "--number-of-temperature-nodes",
            "3",
            "--max-interpolation-error-in-noise",
            "0.02",
            "--overwrite",
        ]
    )
    assert prepare.handler == benchmark.prepare
    assert prepare.number_of_layers == 4 and prepare.overwrite
    run = parser.parse_args(
        [
            "run",
            "--method",
            "premodit",
            "--output-dir",
            str(tmp_path),
            "--num-warmup",
            "20",
            "--num-samples",
            "30",
            "--seed",
            "9",
            "--gradient-repetitions",
            "2",
        ]
    )
    assert (run.num_warmup, run.num_samples, run.seed, run.gradient_repetitions) == (
        20,
        30,
        9,
        2,
    )
    assert run.run_id is None
    assert run.allow_code_revision is False
    assert (
        parser.parse_args(
            ["run", "--method", "diffgrid", "--allow-code-revision"]
        ).allow_code_revision
        is True
    )
    assert parser.parse_args(["run", "--method", "diffgrid"]).num_samples == 1000
    assert (
        parser.parse_args(["summarize", "--output-dir", str(tmp_path)]).run_id is None
    )
    assert (
        parser.parse_args(["run", "--method", "diffgrid", "--run-id", "before"]).run_id
        == "before"
    )


@pytest.mark.parametrize(
    "run_id", ["", " ", ".", "..", "../other", "a/b", "a\\b", "a\0b"]
)
def test_run_id_rejects_unsafe_path_components(storage, run_id):
    with pytest.raises(ValueError, match="Run ID"):
        storage.validate_run_id(run_id)


def test_named_run_isolation_collision_and_legacy_paths(storage, tmp_path):
    first = storage.reserve_result(tmp_path, "diffgrid", "before")
    first["result"].write_text("keep this result")
    second = storage.reserve_result(tmp_path, "diffgrid", "after")
    peer = storage.reserve_result(tmp_path, "premodit", "before")
    assert first["result"] == tmp_path / "runs/before/diffgrid/result.json"
    assert len({first["result"], second["result"], peer["result"]}) == 3
    with pytest.raises(FileExistsError):
        storage.reserve_result(tmp_path, "diffgrid", "before")
    assert first["result"].read_text() == "keep this result"
    legacy = storage.reserve_result(tmp_path, "diffgrid")
    assert legacy == {
        "result": tmp_path / "diffgrid.json",
        "samples": tmp_path / "diffgrid_samples.npz",
    }
    assert storage.reserve_result(tmp_path, "diffgrid") == legacy


@pytest.mark.parametrize("version", [0, 3, "2", True, None])
def test_unknown_schema_is_rejected(storage, tmp_path, version):
    path = tmp_path / "result.json"
    storage.write_json(path, {"schema_version": version, "status": "completed"})
    with pytest.raises(ValueError, match="schema version"):
        storage.read_metadata(path)


@pytest.mark.parametrize("status", ["partial", "failed", None])
def test_unfinished_result_is_rejected(storage, tmp_path, status):
    path = tmp_path / "result.json"
    storage.write_json(path, {"schema_version": 2, "status": status})
    with pytest.raises(ValueError, match="not completed"):
        storage.read_metadata(path)


def test_legacy_metadata_and_case_do_not_invent_provenance(
    benchmark, storage, prepared, tmp_path
):
    paths, original, _ = prepared
    _, metadata, case, digest = benchmark._load_case(tmp_path)
    with case:
        np.testing.assert_array_equal(case["observed_flux"], [1.0, 1.1])
    assert digest == original["artifacts"]["case_sha256"]
    assert metadata["provenance"] is None and metadata["status"] is None
    assert metadata["samples"] is None
    assert set(metadata["legacy_missing_fields"]) == {"provenance", "status", "samples"}
    assert json.loads(paths["prepare"].read_text()) == json.loads(json.dumps(original))
    benchmark._validate_artifacts(paths, metadata, ("premodit", "diffgrid"))


@pytest.mark.parametrize(
    "artifact",
    ["case", "premodit", "premodit_metadata", "diffgrid", "diffgrid_metadata", "cia"],
)
def test_modified_prepared_inputs_are_rejected(benchmark, prepared, tmp_path, artifact):
    paths, metadata, cia_path = prepared
    path = cia_path if artifact == "cia" else paths[artifact]
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="digest|hash"):
        if artifact == "case":
            benchmark._load_case(tmp_path)
        else:
            benchmark._validate_artifacts(paths, metadata, ("premodit", "diffgrid"))


def test_revision_opacity_load_relaxes_version_but_preserves_dtype_and_schema(
    benchmark, storage, tmp_path
):
    import jax.numpy as jnp

    teacher = SimpleNamespace(
        method="synthetic",
        ready=True,
        nu_grid=np.array([1000.0, 1001.0]),
        xsmatrix=lambda temperature, pressure: jnp.broadcast_to(
            1.0e-22 * jnp.exp(-500.0 / temperature[:, None]), (len(pressure), 2)
        ),
    )
    original = benchmark.OpaDiffgrid(
        teacher,
        np.array([700.0, 1000.0, 1500.0]),
        np.array([0.1]),
        min_cross_section=1.0e-30,
    )
    archive_path = tmp_path / "diffgrid.npz"
    benchmark.saveopa(original, str(archive_path), format="npz")
    metadata_path = tmp_path / "diffgrid_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["exojax_version"] = "0.0.external"
    storage.write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="strict=False|version"):
        benchmark._load_opacity("diffgrid", archive_path)
    restored = benchmark._load_opacity(
        "diffgrid", archive_path, allow_code_revision=True
    )
    np.testing.assert_array_equal(
        restored.xsmatrix(jnp.array([800.0])), original.xsmatrix(jnp.array([800.0]))
    )
    previous_x64 = benchmark.config.jax_enable_x64
    try:
        benchmark.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="dtype|precision"):
            benchmark._load_opacity("diffgrid", archive_path, allow_code_revision=True)
    finally:
        benchmark.config.update("jax_enable_x64", previous_x64)
    metadata["schema_version"] = "ioopa@999"
    storage.write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="schema"):
        benchmark._load_opacity("diffgrid", archive_path, allow_code_revision=True)


def test_raw_chains_preserve_order_values_dtype_and_recompute_summary(
    storage, tmp_path
):
    samples = {
        "radius": np.arange(12, dtype=np.float64).reshape(2, 6),
        "vector": np.arange(24, dtype=np.float32).reshape(2, 6, 2),
    }
    extra = {
        "num_steps": np.arange(12, dtype=np.int32).reshape(2, 6),
        "diverging": np.array([[True, False] * 3] * 2),
    }
    path = tmp_path / "samples.npz"
    manifest = storage.save_samples(path, samples, extra, ["vector", "radius"])
    assert manifest["chain_shape"] == [2, 6]
    assert manifest["parameter_order"] == ["vector", "radius"]
    assert manifest["sha256"] == storage.sha256(path)
    restored, fields = storage.load_samples(path, manifest)
    for original, loaded in ((samples, restored), (extra, fields)):
        for name, value in original.items():
            np.testing.assert_array_equal(loaded[name], value)
            assert loaded[name].dtype == value.dtype
    assert int(fields["num_steps"].sum()) == 66
    assert int(fields["diverging"].sum()) == 6
    np.testing.assert_array_equal(restored["radius"].mean(axis=1), [2.5, 8.5])
    for field, replacement in (("shape", [1, 12]), ("dtype", "<f4")):
        invalid = copy.deepcopy(manifest)
        invalid["samples"]["radius"][field] = replacement
        with pytest.raises(ValueError, match="shape or dtype"):
            storage.load_samples(path, invalid)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        storage.load_samples(path, manifest)


@pytest.mark.parametrize(
    "bad_array",
    [
        np.array([[object()]], dtype=object),
        np.ones(4),
        np.ones((1, 3)),
        np.ones((0, 4)),
    ],
)
def test_invalid_raw_arrays_do_not_replace_an_archive(storage, tmp_path, bad_array):
    path = tmp_path / "samples.npz"
    path.write_bytes(b"existing archive")
    with pytest.raises(ValueError):
        storage.save_samples(
            path, {"radius": np.ones((1, 4)), "bad": bad_array}, {}, ["radius", "bad"]
        )
    assert path.read_bytes() == b"existing archive"


def test_object_archive_is_rejected_even_with_matching_digest(storage, tmp_path):
    path = tmp_path / "samples.npz"
    manifest = storage.save_samples(path, {"radius": np.ones((1, 4))}, {}, ["radius"])
    key = manifest["samples"]["radius"]["archive_key"]
    np.savez(path, **{key: np.full((1, 4), "unsafe", dtype=object)})
    manifest["sha256"] = storage.sha256(path)
    with pytest.raises(ValueError, match="Object arrays"):
        storage.load_samples(path, manifest)


def _result(metadata, method="diffgrid", run_id="before"):
    return {
        "schema_version": 2,
        "status": "completed",
        "method": method,
        "run_id": run_id,
        "case_sha256": metadata["artifacts"]["case_sha256"],
        "run": {"num_chains": 1, "num_samples": 4, "num_warmup": 2, "seed": 0},
        "environment": {
            "exojax": "before",
            "jax": "test",
            "jax_enable_x64": True,
            "device_kind": "cpu",
            "device_platform": "cpu",
        },
        "physics": {"noise_sigma": 0.05, "temperature_clip": [400.0, 1500.0]},
        "provenance": {
            "environment": {"JAX_PLATFORMS": "cpu"},
            "dependencies": {"exojax": "before", "numpy": "test"},
            "code_sha256": "before",
            "git": {"commit": "before"},
        },
        "timings": {
            "opacity_load_seconds": 0.1,
            "model_setup_seconds": 0.1,
            "compile_and_warmup_seconds": 1.0,
            "sampling_compile_and_run_seconds": 2.0,
            "sampling_seconds_per_sample": 0.5,
            "cold_milliseconds_per_leapfrog_step": 50.0,
        },
        "potential_gradient_benchmark": {"median_evaluation_seconds": 0.1},
        "diagnostics": {
            "total_num_steps": 999,
            "minimum_effective_sample_size_per_second": None,
            "number_of_divergences": 0,
            "mean_accept_probability": 0.9,
        },
        "device_memory": {},
        "host_peak_rss_bytes": None,
    }


def test_revision_comparison_permits_only_code_changes(benchmark, prepared):
    _, metadata, _ = prepared
    baseline = _result(metadata)
    candidate = copy.deepcopy(baseline)
    candidate["environment"]["exojax"] = "after"
    candidate["provenance"]["dependencies"]["exojax"] = "after"
    candidate["provenance"]["code_sha256"] = "after"
    candidate["provenance"]["git"]["commit"] = "after"
    benchmark._validate_results(metadata, [baseline, baseline])
    benchmark._validate_results(
        metadata, [baseline, candidate], revision_comparison=True
    )
    with pytest.raises(ValueError, match="environment|code|commit"):
        benchmark._validate_results(metadata, [baseline, candidate])
    for section, key, value in (
        ("environment", "device_kind", "different CPU"),
        ("environment", "jax_enable_x64", False),
        ("run", "seed", 2),
        ("physics", "noise_sigma", 0.1),
    ):
        invalid = copy.deepcopy(candidate)
        invalid[section][key] = value
        with pytest.raises(ValueError):
            benchmark._validate_results(
                metadata, [baseline, invalid], revision_comparison=True
            )
    for section in ("dependencies", "environment"):
        invalid = copy.deepcopy(candidate)
        invalid["provenance"][section][
            "numpy" if section == "dependencies" else "JAX_PLATFORMS"
        ] = "changed"
        with pytest.raises(ValueError, match="provenance"):
            benchmark._validate_results(
                metadata, [baseline, invalid], revision_comparison=True
            )
    for key, value in (
        ("case_sha256", "changed"),
        ("status", "partial"),
        ("schema_version", 99),
    ):
        invalid = dict(candidate, **{key: value})
        with pytest.raises(ValueError):
            benchmark._validate_results(
                metadata, [baseline, invalid], revision_comparison=True
            )


@pytest.mark.parametrize("run_id", [None, "before"])
def test_summary_explicitly_selects_legacy_or_named_results(
    benchmark, storage, prepared, tmp_path, monkeypatch, run_id
):
    paths, metadata, _ = prepared
    metadata.update(
        timings={"diffgrid_build_seconds": 1.0},
        diffgrid={
            "table_payload_bytes": 64,
            "maximum_interpolation_error_in_noise": 0.001,
        },
    )
    storage.write_json(paths["prepare"], metadata)
    prepare_digest = storage.sha256(paths["prepare"])
    monkeypatch.setattr(
        benchmark, "_minimum_effective_sample_size", lambda samples: None
    )
    monkeypatch.setattr(benchmark, "_plot_comparison", lambda *args: None)
    for selected in (None, "before"):
        for method in ("premodit", "diffgrid"):
            saved = storage.reserve_result(tmp_path, method, selected)
            result = _result(metadata, method, selected)
            result["prepare_sha256"] = prepare_digest
            result["samples"] = storage.save_samples(
                saved["samples"],
                {"radius": np.ones((1, 4))},
                {
                    "num_steps": np.array([[1, 2, 3, 4]]),
                    "diverging": np.zeros((1, 4), dtype=bool),
                    "accept_prob": np.full((1, 4), 0.75),
                },
                ["radius"],
            )
            if selected is None:
                result["schema_version"] = 1
                result.pop("samples")
                result.pop("run_id")
            if method == "premodit":
                result["timings"]["sampling_compile_and_run_seconds"] = (
                    8.0 if selected is None else 4.0
                )
            storage.write_json(saved["result"], result)
    args = ["summarize", "--output-dir", str(tmp_path)]
    if run_id is not None:
        args += ["--run-id", run_id]
    benchmark.summarize_results(benchmark._parser().parse_args(args))
    directory = tmp_path if run_id is None else tmp_path / "runs" / run_id
    comparison = json.loads((directory / "comparison.json").read_text())
    assert comparison["run_id"] == run_id
    assert comparison["sampling_speedup_premodit_over_diffgrid"] == (
        4.0 if run_id is None else 2.0
    )
    assert (directory / "comparison.csv").exists()
    if run_id is not None:
        assert comparison["methods"]["diffgrid"]["diagnostics"]["total_num_steps"] == 10
        result_path = storage.result_paths(tmp_path, "diffgrid", run_id)["result"]
        changed = json.loads(result_path.read_text())
        changed["prepare_sha256"] = "changed"
        storage.write_json(result_path, changed)
        with pytest.raises(ValueError, match="prepare.json digest"):
            benchmark.summarize_results(benchmark._parser().parse_args(args))


@pytest.mark.parametrize("failure", [False, True])
def test_run_persists_partial_then_completion_or_failure(
    benchmark, storage, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(benchmark, "_provenance", lambda *args: {})

    def execute(args, state, saved):
        assert json.loads(saved["result"].read_text())["status"] == "partial"
        state["stage"] = "sampling"
        if failure:
            raise RuntimeError("synthetic sampler failure")

    monkeypatch.setattr(benchmark, "_run_method", execute)
    args = benchmark._parser().parse_args(
        [
            "run",
            "--output-dir",
            str(tmp_path),
            "--method",
            "diffgrid",
            "--run-id",
            "test",
        ]
    )
    if failure:
        with pytest.raises(RuntimeError, match="synthetic sampler failure"):
            benchmark.run_method(args)
    else:
        benchmark.run_method(args)
    result = json.loads(
        storage.result_paths(tmp_path, "diffgrid", "test")["result"].read_text()
    )
    assert result["status"] == ("failed" if failure else "completed")
    if failure:
        assert result["failure"] == {
            "stage": "sampling",
            "type": "RuntimeError",
            "message": "synthetic sampler failure",
        }


def test_provenance_records_dirty_and_untracked_execution_inputs(
    storage, tmp_path, monkeypatch
):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n")
    git("add", "tracked.py")
    git(
        "-c",
        "user.name=Offline Test",
        "-c",
        "user.email=offline@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Fixture",
    )
    clean = storage.collect_provenance(tmp_path, [tracked], {"seed": 0})
    tracked.write_text("value = 2\n")
    untracked = tmp_path / "executed.py"
    untracked.write_text("value = 3\n")
    missing = tmp_path / "missing.npz"
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    dirty = storage.collect_provenance(
        tmp_path, [tracked, untracked, missing], {"seed": 0}
    )
    assert dirty["git"]["commit"] == git("rev-parse", "HEAD")
    assert dirty["git"]["dirty"] is True
    assert dirty["git"]["fixed_head_reproducible"] is False
    assert dirty["git"]["tracked_diff_sha256"] != clean["git"]["tracked_diff_sha256"]
    assert dirty["code_sha256"] != clean["code_sha256"]
    assert dirty["settings_sha256"] == clean["settings_sha256"]
    files = {entry["path"]: entry for entry in dirty["files"]}
    assert files["executed.py"]["tracked"] is False
    assert files["executed.py"]["sha256"] == storage.sha256(untracked)
    assert (
        files["missing.npz"]["sha256"] is None
        and files["missing.npz"]["unavailable_reason"]
    )
    assert dirty["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "numpy" in dirty["dependencies"]


def test_fixed_saved_case_matches_pre_pr1_forward_and_likelihood(
    benchmark, storage, tmp_path
):
    """Snapshot from ce42b1d: synthetic opacity, actual RT, rotation and LSF."""
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    case_config = benchmark.CaseConfig(
        number_of_layers=4,
        number_of_wavenumbers=256,
        number_of_observed_wavenumbers=8,
    )
    nu_data, wavelength_data = benchmark._observation_grid(case_config)
    nu_grid, _, resolution = benchmark._model_grid(wavelength_data, case_config)
    case_path = tmp_path / "case.npz"
    observed_flux = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    storage.write_npz(
        case_path, nu_grid=nu_grid, nu_data=nu_data, observed_flux=observed_flux
    )
    with np.load(case_path, allow_pickle=False) as case:
        nu_grid, nu_data, observed_flux = (
            case[key] for key in ("nu_grid", "nu_data", "observed_flux")
        )
    art = benchmark.ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=case_config.pressure_top,
        pressure_btm=case_config.pressure_bottom,
        nlayer=case_config.number_of_layers,
    )
    art.change_temperature_range(
        case_config.temperature_min, case_config.temperature_max
    )

    def cross_section(temperature, pressure):
        shape = 1.0 + 0.4 * jnp.exp(-(((jnp.asarray(nu_grid) - 6080.0) / 12.0) ** 2))
        return (
            2.0e-23
            * (temperature[:, None] / 1000.0) ** 0.4
            * (1.0 + 0.2 * pressure[:, None])
            * shape
        )

    context = {
        "art": art,
        "opa_cia": SimpleNamespace(
            logacia_matrix=lambda temperature: jnp.full(
                (temperature.size, nu_grid.size), -44.0
            )
        ),
        "nu_data": jnp.asarray(nu_data),
        "nu_grid": jnp.asarray(nu_grid),
        "velocity_array": benchmark.velocity_grid(
            resolution, case_config.maximum_vsini
        ),
        "instrument_beta": benchmark.resolution_to_gaussian_std(
            case_config.instrument_resolution
        ),
        "hydrogen_volume_mixing_ratio": case_config.hydrogen_mass_mixing_ratio
        * case_config.mean_molecular_weight
        / benchmark.molinfo.molmass_isotope("H2"),
    }
    expected = [
        1.0490610829314018,
        1.0441744273106561,
        1.034320460147855,
        1.002089473723343,
        0.973055226105466,
        0.9962347598799349,
        1.017008446899888,
        1.0165965201177547,
    ]
    temperature = art.powerlaw_temperature(
        benchmark.TRUTH["temperature_at_1bar"], benchmark.TRUTH["temperature_index"]
    )
    for method in ("premodit", "diffgrid"):
        opacity = SimpleNamespace(
            method=method,
            molmass=16.04,
            xsmatrix=(
                cross_section
                if method == "premodit"
                else lambda temperature: cross_section(temperature, art.pressure)
            ),
        )
        forward = benchmark._make_forward_model(opacity, context, case_config)
        prediction = forward(
            temperature,
            *(
                benchmark.TRUTH[key]
                for key in (
                    "methane_mass_mixing_ratio",
                    "radius",
                    "radial_velocity",
                    "vsini",
                )
            ),
        )
        np.testing.assert_allclose(prediction, expected, rtol=2.0e-12, atol=1.0e-12)
        log_likelihood = norm.logpdf(
            observed_flux, prediction, case_config.noise_sigma
        ).sum()
        np.testing.assert_allclose(log_likelihood, -23.794705770640917, rtol=2.0e-12)
