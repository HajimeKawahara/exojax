"""CLI smoke tests for the WASP-39b full JWST spectra example."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "wasp39b_full_jwst_spectra.py"


def run_example(*args, env_overrides=None):
    """Run the example script in a subprocess and capture text output."""
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_synthetic_exomolop_ckd(
    root,
    nu_min=1800.0,
    nu_max=3900.0,
    n_band=25,
    molecule="H2O",
    mol_mass=18.0,
    samples=None,
    weights=None,
):
    """Write a minimal ExoMolOP-style CKD h5 table."""
    relative_paths = {
        "H2O": ("H2O", "1H2-16O", "POKAZATEL"),
        "CO2": ("CO2", "12C-16O2", "UCL-4000"),
    }
    table_dir = root.joinpath(*relative_paths.get(molecule, (molecule, "synthetic")))
    table_dir.mkdir(parents=True)
    table_path = table_dir / f"synthetic_{molecule.lower()}.h5"

    n_pressure = 2
    n_temperature = 2
    samples = np.array([0.2, 0.5, 0.8]) if samples is None else np.asarray(samples)
    weights = (
        np.array([0.25, 0.5, 0.25]) if weights is None else np.asarray(weights)
    )
    n_g = samples.size
    nu_centers = np.linspace(nu_min, nu_max, n_band)
    temperatures = np.array([500.0, 2000.0])
    pressures = np.array([1.0e-12, 1.0e2])
    base_xs = 1.0e-27 * (1.0 + 0.2 * np.sin(nu_centers / 350.0))

    kcoeff = np.empty((n_pressure, n_temperature, n_band, n_g))
    for ipressure in range(n_pressure):
        for itemperature in range(n_temperature):
            for iband in range(n_band):
                for ig in range(n_g):
                    kcoeff[ipressure, itemperature, iband, ig] = base_xs[iband] * (
                        1.0 + 0.2 * ipressure + 0.1 * itemperature + 0.05 * ig
                    )

    with h5py.File(table_path, "w") as handle:
        handle.create_dataset("mol_name", data=np.array([molecule.encode()]))
        handle.create_dataset("mol_mass", data=np.array([mol_mass]))
        handle.create_dataset("bin_centers", data=nu_centers)
        handle.create_dataset("samples", data=samples)
        handle.create_dataset("weights", data=weights)
        handle.create_dataset("t", data=temperatures)
        handle.create_dataset("p", data=pressures)
        handle.create_dataset("kcoeff", data=kcoeff)

    return table_path


def create_minimal_opacity_root(root):
    """Create a minimal opacity root containing the H2-H2 CIA placeholder."""
    cia_dir = root / ".db_CIA"
    cia_dir.mkdir(parents=True)
    (cia_dir / "H2-H2_2011.cia").touch()
    return root


def create_tracked_cia_opacity_root(root):
    """Create an opacity root with the tracked compact H2-H2 CIA test file."""
    cia_dir = root / ".db_CIA"
    cia_dir.mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "src" / "exojax" / "data" / "testdata" / "H2-H2_TEST.cia",
        cia_dir / "H2-H2_2011.cia",
    )
    return root


def create_minimal_wasp39_data_dir(root, *, nirspec_wavelength=None, nirspec_std=None):
    """Create minimal observed-data files accepted by the example loader."""
    root.mkdir(parents=True)
    niriss = np.array(
        [
            [1000.0, 0.145, 0.001, 0.001],
            [1100.0, 0.146, 0.001, 0.001],
        ]
    )
    np.savetxt(root / "niriss_order1.txt", niriss)
    np.savetxt(root / "niriss_order2.txt", niriss + np.array([200.0, 0.0, 0.0, 0.0]))
    if nirspec_wavelength is None:
        nirspec_wavelength = np.array([3000.0, 3100.0])
    np.save(root / "wavelength.npy", np.asarray(nirspec_wavelength))
    np.save(root / "wasp39b_nirspec_g395h_rp_mean.npy", np.array([0.145, 0.146]))
    if nirspec_std is None:
        nirspec_std = np.array([0.001, 0.001])
    np.save(root / "wasp39b_nirspec_g395h_rp_std.npy", np.asarray(nirspec_std))
    with h5py.File(root / "miri.h5", "w") as handle:
        handle.create_dataset("wavelength", data=np.array([5.0, 5.1]))
        handle.create_dataset("dppm", data=np.array([21000.0, 21100.0]))
        handle.create_dataset("dppm_error", data=np.array([100.0, 100.0]))
    return root


def test_wide_premodit_reports_cli_error_without_traceback():
    result = run_example(
        "--data-mode",
        "wide",
        "--opacity-mode",
        "premodit",
        "--channels",
        "nirspec_g395h,miri_lrs",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "H2H2",
        "--check-forward",
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 2
    assert "Wide-wavelength retrieval requires --opacity-mode ckd" in result.stderr
    assert "Traceback" not in result.stderr


def test_wide_ckd_summarize_data_reports_combined_channels():
    result = run_example(
        "--summarize-data",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h,miri_lrs",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "H2H2",
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 0
    assert "nirspec_g395h: n=3328" in result.stdout
    assert "miri_lrs: n=28" in result.stdout
    assert "combined: n=3356" in result.stdout
    assert "Selected molecules: H2O, CO2" in result.stdout


def test_invalid_max_observed_reports_cli_error_without_traceback():
    result = run_example(
        "--summarize-data",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--max-observed",
        "0",
    )

    assert result.returncode == 2
    assert "--max-observed must be a positive integer when set" in result.stderr
    assert "Traceback" not in result.stderr


def test_duplicate_cli_selections_report_errors_without_traceback():
    cases = [
        ("--molecules", "H2O,H2O", "Duplicate molecules in --molecules: H2O"),
        ("--channels", "nirspec_g395h,nirspec_g395h", "Duplicate channels in --channels: nirspec_g395h"),
        ("--cia-pairs", "H2H2,H2H2", "Duplicate CIA pairs in --cia-pairs: H2H2"),
    ]
    for option, value, message in cases:
        result = run_example(
            "--summarize-data",
            "--data-mode",
            "wide",
            "--opacity-mode",
            "ckd",
            "--channels",
            "nirspec_g395h",
            "--molecules",
            "H2O",
            "--cia-pairs",
            "none",
            option,
            value,
        )

        assert result.returncode == 2
        assert message in result.stderr
        assert "Traceback" not in result.stderr


def test_observed_data_validation_rejects_nonpositive_uncertainty(tmp_path):
    data_dir = create_minimal_wasp39_data_dir(
        tmp_path / "wasp39_data", nirspec_std=np.array([0.001, 0.0])
    )

    result = run_example(
        "--summarize-data",
        "--data-mode",
        "nirspec",
        "--opacity-mode",
        "premodit",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--data-dir",
        str(data_dir),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 1
    assert "selected observed data uncertainties must be positive" in result.stderr


def test_status_artifact_paths_require_matching_check_modes(tmp_path):
    input_status_json = tmp_path / "input_status.json"
    forward_check_json = tmp_path / "forward_check.json"

    input_result = run_example(
        "--summarize-data",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--input-status-json",
        str(input_status_json),
    )
    assert input_result.returncode == 2
    assert "--input-status-json requires --check-inputs" in input_result.stderr
    assert "Traceback" not in input_result.stderr

    forward_result = run_example(
        "--summarize-data",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--forward-check-json",
        str(forward_check_json),
    )
    assert forward_result.returncode == 2
    assert "--forward-check-json requires --check-forward" in forward_result.stderr
    assert "Traceback" not in forward_result.stderr


def test_ckd_only_options_require_ckd_opacity_mode():
    cases = [
        (
            ["--ckd-table-paths", "H2O=/tmp/table.h5"],
            "--ckd-table-paths requires --opacity-mode ckd",
        ),
        (
            ["--allow-ckd-download"],
            "--allow-ckd-download requires --opacity-mode ckd",
        ),
    ]
    for extra_args, message in cases:
        result = run_example(
            "--summarize-data",
            "--data-mode",
            "nirspec",
            "--opacity-mode",
            "premodit",
            "--channels",
            "nirspec_g395h",
            "--molecules",
            "H2O",
            "--cia-pairs",
            "none",
            *extra_args,
        )

        assert result.returncode == 2
        assert message in result.stderr
        assert "Traceback" not in result.stderr


def test_jax_platform_does_not_override_existing_environment():
    result = run_example(
        "--summarize-data",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--jax-platform",
        "cpu",
        env_overrides={"JAX_PLATFORMS": "gpu"},
    )

    assert result.returncode == 0
    assert "JAX_PLATFORMS is already set to gpu" in result.stdout
    assert "--jax-platform cpu will not override it" in result.stdout


def test_plot_data_only_uses_writable_matplotlib_cache(tmp_path):
    plot_path = tmp_path / "plot.png"

    result = run_example(
        "--plot-data-only",
        "--channels",
        "nirspec_g395h",
        "--data-plot-path",
        str(plot_path),
        env_overrides={"HOME": "/proc"},
    )

    assert result.returncode == 0, result.stderr
    assert plot_path.exists()
    assert "Matplotlib created a temporary cache directory" not in result.stderr


def test_ckd_check_forward_runs_with_synthetic_table_without_cia(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 0, result.stderr
    assert "synthetic_h2o.h5" in result.stdout
    assert "Jax plugin configuration error" not in result.stderr
    assert "Both input wavelength and output wavenumber are in ascending order" not in result.stderr
    assert "Forward-model check:" in result.stdout
    assert "  CIA pairs: (none)" in result.stdout
    assert "  observed shape: (64,)" in result.stdout
    assert "  model shape: (64,)" in result.stdout
    assert "  finite model: True" in result.stdout
    assert "  CKD bands: n=" in result.stdout
    assert "  CKD band edges:" in result.stdout


def test_ckd_check_forward_runs_with_tracked_cia_testdata(tmp_path):
    data_dir = create_minimal_wasp39_data_dir(
        tmp_path / "wasp39_data",
        nirspec_wavelength=np.array([1.0e7 / 4320.0, 1.0e7 / 4360.0]),
    )
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(
        ckd_root,
        nu_min=4200.0,
        nu_max=4400.0,
        n_band=16,
    )
    opacity_root = create_tracked_cia_opacity_root(tmp_path / "opacity")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--data-dir",
        str(data_dir),
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--skip-corner",
    )

    assert result.returncode == 0, result.stderr
    assert "Both input wavelength and output wavenumber are in ascending order" not in result.stderr
    assert "Load CIA:  H2-H2" in result.stdout
    assert "  CIA pairs: H2H2" in result.stdout
    assert "  observed shape: (2,)" in result.stdout
    assert "  model shape: (2,)" in result.stdout
    assert "  finite model: True" in result.stdout


def test_ckd_check_forward_runs_with_synthetic_wide_channels(tmp_path):
    ckd_root = tmp_path / "ckd"
    forward_json = tmp_path / "diagnostics" / "forward" / "forward_check.json"
    write_synthetic_exomolop_ckd(
        ckd_root,
        nu_min=750.0,
        nu_max=17000.0,
        n_band=48,
    )

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "niriss_order1,niriss_order2,nirspec_g395h,miri_lrs",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "96",
        "--forward-check-json",
        str(forward_json),
        "--skip-corner",
    )

    assert result.returncode == 0, result.stderr
    assert "Forward-model check:" in result.stdout
    assert "  channels: niriss_order1, niriss_order2, nirspec_g395h, miri_lrs" in result.stdout
    assert "  fiducial RV: -83.180 km/s" in result.stdout
    assert "  observed shape: (96,)" in result.stdout
    assert "  model shape: (96,)" in result.stdout
    assert "  finite model: True" in result.stdout
    assert "  CKD bands: n=" in result.stdout
    assert f"Forward-check JSON saved to {forward_json}" in result.stdout
    assert forward_json.exists()
    with open(forward_json) as handle:
        forward_status = json.load(handle)
    assert forward_status["shape_matches"] is True
    assert forward_status["finite_model"] is True
    assert forward_status["observed_shape"] == [96]
    assert forward_status["model_shape"] == [96]
    assert forward_status["rv_fixed_kms"] == -83.18
    assert forward_status["fiducial_rv_kms"] == -83.18
    assert forward_status["ckd_sources"]["H2O"].endswith("synthetic_h2o.h5")
    assert forward_status["input_status"]["ready_for_local_run"] is True
    assert forward_status["input_status"]["problems"] == []
    assert (
        forward_status["input_status"]["ckd_sources"]["H2O"]["resolved_table"]
        == forward_status["ckd_sources"]["H2O"]
    )
    table_summary = forward_status["ckd_table_summary"]["H2O"]
    assert table_summary["molmass"] == 18.0
    assert table_summary["n_temperature"] == 2
    assert table_summary["n_pressure"] == 2
    assert table_summary["n_g"] == 3
    assert table_summary["weight_sum"] == 1.0
    assert table_summary["source"].endswith("synthetic_h2o.h5")
    assert sum(
        item["n_selected"]
        for item in forward_status["retrieval_channel_summary"].values()
    ) == 96


def test_ckd_check_forward_runs_with_matching_h2o_co2_tables(tmp_path):
    ckd_root = tmp_path / "ckd"
    forward_json = tmp_path / "forward_check_h2o_co2.json"
    table_h2o = write_synthetic_exomolop_ckd(ckd_root)
    table_co2 = write_synthetic_exomolop_ckd(
        ckd_root,
        molecule="CO2",
        mol_mass=44.0,
    )

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--forward-check-json",
        str(forward_json),
        "--skip-corner",
    )

    assert result.returncode == 0, result.stderr
    assert "  molecules: H2O, CO2" in result.stdout
    assert "  observed shape: (64,)" in result.stdout
    assert "  model shape: (64,)" in result.stdout
    assert "  finite model: True" in result.stdout
    with open(forward_json) as handle:
        forward_status = json.load(handle)
    assert forward_status["molecules"] == ["H2O", "CO2"]
    assert forward_status["ckd_sources"]["H2O"] == str(table_h2o)
    assert forward_status["ckd_sources"]["CO2"] == str(table_co2)
    assert forward_status["input_status"]["ready_for_local_run"] is True
    assert forward_status["input_status"]["problems"] == []
    assert forward_status["ckd_table_summary"]["H2O"]["molmass"] == 18.0
    assert forward_status["ckd_table_summary"]["CO2"]["molmass"] == 44.0
    assert forward_status["ckd_table_summary"]["H2O"]["n_g"] == 3
    assert forward_status["ckd_table_summary"]["CO2"]["n_g"] == 3


def test_ckd_check_forward_rejects_mismatched_quadrature_weights(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    write_synthetic_exomolop_ckd(
        ckd_root,
        molecule="CO2",
        mol_mass=44.0,
        weights=np.array([0.2, 0.3, 0.5]),
    )

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "CKD table quadrature weights mismatch for CO2" in result.stderr
    assert "H2O source=" in result.stderr
    assert "CO2 source=" in result.stderr
    assert "shape=(3,)" in result.stderr


def test_ckd_check_forward_rejects_mismatched_band_centers(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    write_synthetic_exomolop_ckd(
        ckd_root,
        molecule="CO2",
        mol_mass=44.0,
        nu_min=1810.0,
        nu_max=3910.0,
    )

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "CKD table band centers mismatch for CO2" in result.stderr
    assert "H2O source=" in result.stderr
    assert "CO2 source=" in result.stderr


def test_ckd_check_forward_rejects_mismatched_g_grid(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    write_synthetic_exomolop_ckd(
        ckd_root,
        molecule="CO2",
        mol_mass=44.0,
        samples=np.array([0.1, 0.5, 0.9]),
    )

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "CKD table g-grid mismatch for CO2" in result.stderr
    assert "H2O source=" in result.stderr
    assert "CO2 source=" in result.stderr


def test_check_inputs_reports_multiple_ckd_h5_tables(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "diagnostics" / "inputs" / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert "H2O h5 table:" in result.stdout
    assert "multiple (2 non-empty); specify a directory with one table" in result.stdout
    assert status_json.exists()
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["table_status"] == "multiple"
    assert len(status["ckd_sources"]["H2O"]["h5_candidates"]) == 2
    assert status["problems"] == ["ckd_sources.H2O.table: multiple"]


def test_check_inputs_json_records_resolved_default_ckd_table(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_path = write_synthetic_exomolop_ckd(ckd_root)
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    source = status["ckd_sources"]["H2O"]
    assert status["ready_for_local_run"] is True
    assert source["explicit_table"] is False
    assert source["table_status"] == "ok"
    assert source["resolved_table"] == str(table_path)
    assert source["resolved_table_metadata"]["size_bytes"] == table_path.stat().st_size
    assert source["resolved_table_metadata"]["mtime_ns"] == table_path.stat().st_mtime_ns
    assert source["table_schema"]["status"] == "ok"
    assert source["table_schema"]["datasets"]["kcoeff"]["shape"] == [2, 2, 25, 3]
    assert source["table_schema"]["numeric_summary"]["weights"]["min"] == 0.25
    assert source["table_schema"]["numeric_summary"]["weights"]["max"] == 0.5


def test_check_inputs_json_allows_ckd_download_without_local_table(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O,CO2",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--allow-ckd-download",
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert "H2O download candidates:" in result.stdout
    assert (
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in result.stdout
    )
    assert "CO2 download candidates:" in result.stdout
    assert (
        "12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in result.stdout
    )
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["opacity_mode"] == "ckd"
    assert status["selections"]["allow_ckd_download"] is True
    assert status["ckd_sources"]["H2O"]["table_status"] == "not_checked"
    assert status["ckd_sources"]["CO2"]["table_status"] == "not_checked"
    assert status["ckd_sources"]["H2O"]["download_required"] is True
    assert status["ckd_sources"]["CO2"]["download_required"] is True
    assert status["ckd_sources"]["H2O"]["download_target"].endswith(
        "H2O/1H2-16O/POKAZATEL"
    )
    assert status["ckd_sources"]["CO2"]["download_target"].endswith(
        "CO2/12C-16O2/UCL-4000"
    )
    h2o_candidate_tables = status["ckd_sources"]["H2O"]["download_candidate_tables"]
    co2_candidate_tables = status["ckd_sources"]["CO2"]["download_candidate_tables"]
    assert h2o_candidate_tables[0].endswith(
        "H2O/1H2-16O/POKAZATEL/"
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert h2o_candidate_tables[1].endswith(
        "H2O/1H2-16O/POKAZATEL/"
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert co2_candidate_tables[0].endswith(
        "CO2/12C-16O2/UCL-4000/"
        "12C-16O2__UCL-4000__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert co2_candidate_tables[1].endswith(
        "CO2/12C-16O2/UCL-4000/"
        "12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert status["ready_for_local_run"] is True
    assert status["problems"] == []


def test_check_inputs_without_local_table_does_not_import_h5py(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--allow-ckd-download",
        "--jax-platform",
        "cpu",
        env_overrides={"PYTHONPROFILEIMPORTTIME": "1"},
    )

    assert result.returncode == 0
    assert "h5py" not in result.stderr


def test_check_inputs_reports_invalid_nonempty_ckd_h5_schema(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    table_dir.mkdir(parents=True)
    table_path = table_dir / "broken.h5"
    with h5py.File(table_path, "w") as handle:
        handle.create_dataset("kcoeff", data=np.ones((1, 1, 1, 1)))
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert "H2O h5 schema: missing_datasets" in result.stdout
    assert "missing datasets: mol_mass" in result.stdout
    with open(status_json) as handle:
        status = json.load(handle)
    source = status["ckd_sources"]["H2O"]
    assert status["ready_for_local_run"] is False
    assert source["table_status"] == "ok"
    assert source["resolved_table"] == str(table_path)
    assert source["table_schema"]["status"] == "missing_datasets"
    assert "mol_mass" in source["table_schema"]["missing_datasets"]
    assert status["problems"] == ["ckd_sources.H2O.schema: missing_datasets"]


def test_check_inputs_reports_invalid_ckd_h5_numeric_axes(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    table_path = write_synthetic_exomolop_ckd(
        ckd_root, weights=np.array([0.5, 0.5, 0.5])
    )
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert "H2O h5 schema: invalid_values" in result.stdout
    assert "weights must sum to one" in result.stdout
    with open(status_json) as handle:
        status = json.load(handle)
    source = status["ckd_sources"]["H2O"]
    assert status["ready_for_local_run"] is False
    assert source["resolved_table"] == str(table_path)
    assert source["table_schema"]["status"] == "invalid_values"
    assert "weights must sum to one" in source["table_schema"]["value_issues"]
    assert status["problems"] == ["ckd_sources.H2O.schema: invalid_values"]


def test_check_inputs_json_reports_missing_default_ckd_directory(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["directory_status"] == "missing"
    assert status["ckd_sources"]["H2O"]["table_status"] == "not_checked"
    assert "H2O download candidates:" in result.stdout
    assert (
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in result.stdout
    )
    assert status["problems"] == ["ckd_sources.H2O.directory: missing"]


def test_check_inputs_json_still_reports_multiple_ckd_h5_with_download_allowed(
    tmp_path,
):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--allow-ckd-download",
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["allow_ckd_download"] is True
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["table_status"] == "multiple"
    assert status["ckd_sources"]["H2O"]["download_required"] is False
    assert status["problems"] == ["ckd_sources.H2O.table: multiple"]


def test_check_inputs_json_does_not_require_ckd_tables_for_premodit(tmp_path):
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "nirspec",
        "--opacity-mode",
        "premodit",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["opacity_mode"] == "premodit"
    snapshot_status = status["premodit_snapshots"]["H2O"]
    assert any(path.endswith("opa_H2O.zarr") for path in snapshot_status["candidates"])
    assert any(path.endswith("opaH2O.zarr") for path in snapshot_status["candidates"])
    assert not any(problem.startswith("ckd_sources.") for problem in status["problems"])
    if snapshot_status["status"] == "ok":
        assert snapshot_status["resolved_path"].endswith("examples/opaH2O.zarr")
        assert status["ready_for_local_run"] is True
        assert status["problems"] == []
    else:
        assert snapshot_status["status"] == "missing"
        assert status["ready_for_local_run"] is False
        assert status["problems"] == ["premodit_snapshots.H2O: missing"]


def test_check_inputs_reports_empty_default_ckd_table(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    table_dir.mkdir(parents=True)
    empty_table = table_dir / "empty.h5"
    empty_table.touch()
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert f"H2O h5 table: {table_dir}/*.h5 (empty)" in result.stdout
    assert "H2O download candidates:" in result.stdout
    assert (
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in result.stdout
    )
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["table_status"] == "empty"
    assert status["ckd_sources"]["H2O"]["download_required"] is False
    assert status["problems"] == ["ckd_sources.H2O.table: empty"]

    download_status_json = tmp_path / "download_input_status.json"
    download_result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--allow-ckd-download",
        "--input-status-json",
        str(download_status_json),
        "--jax-platform",
        "cpu",
    )

    assert download_result.returncode == 0
    with open(download_status_json) as handle:
        download_status = json.load(handle)
    assert download_status["ready_for_local_run"] is True
    assert download_status["ckd_sources"]["H2O"]["table_status"] == "empty"
    assert download_status["ckd_sources"]["H2O"]["download_required"] is True
    assert download_status["problems"] == []


def test_check_inputs_ignores_empty_ckd_h5_when_valid_table_exists(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_path = write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    empty_table = table_dir / "old_failed_download.h5"
    empty_table.touch()
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert f"H2O h5 table: {table_dir}/*.h5 (ok)" in result.stdout
    with open(status_json) as handle:
        status = json.load(handle)
    source = status["ckd_sources"]["H2O"]
    assert status["ready_for_local_run"] is True
    assert source["table_status"] == "ok"
    assert source["resolved_table"] == str(table_path)
    assert source["ignored_empty_h5_candidates"] == [str(empty_table)]
    assert status["problems"] == []


def test_check_inputs_reports_explicit_ckd_table_override(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_path = write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--ckd-table-paths",
        f"H2O={table_path}",
        "--opacity-root",
        str(opacity_root),
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    assert "H2O directory:" in result.stdout
    assert "not used; explicit table override" in result.stdout
    assert f"H2O h5 table: {table_path} (ok; explicit)" in result.stdout
    assert "multiple (2 non-empty)" not in result.stdout
    assert f"Input status JSON saved to {status_json}" in result.stdout
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["opacity_mode"] == "ckd"
    assert status["ready_for_local_run"] is True
    assert status["problems"] == []
    assert status["ckd_sources"]["H2O"]["table"] == str(table_path)
    assert status["ckd_sources"]["H2O"]["resolved_table"] == str(table_path)
    assert (
        status["ckd_sources"]["H2O"]["resolved_table_metadata"]["size_bytes"]
        == table_path.stat().st_size
    )
    assert status["ckd_sources"]["H2O"]["table_status"] == "ok"
    assert status["ckd_sources"]["H2O"]["directory_status"] == "not_used_explicit_table"


def test_check_inputs_reports_missing_explicit_ckd_table_with_download_allowed(
    tmp_path,
):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    missing_table = tmp_path / "missing.h5"
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--ckd-table-paths",
        f"H2O={missing_table}",
        "--opacity-root",
        str(opacity_root),
        "--allow-ckd-download",
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["allow_ckd_download"] is True
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["explicit_table"] is True
    assert status["ckd_sources"]["H2O"]["table_status"] == "missing"
    assert status["problems"] == ["ckd_sources.H2O.table: missing"]


def test_check_inputs_reports_non_h5_explicit_ckd_table_with_download_allowed(
    tmp_path,
):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"
    table_path = tmp_path / "table.txt"
    table_path.write_text("not an h5 file")
    status_json = tmp_path / "input_status.json"

    result = run_example(
        "--check-inputs",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--ckd-root",
        str(ckd_root),
        "--ckd-table-paths",
        f"H2O={table_path}",
        "--opacity-root",
        str(opacity_root),
        "--allow-ckd-download",
        "--input-status-json",
        str(status_json),
        "--jax-platform",
        "cpu",
    )

    assert result.returncode == 0
    with open(status_json) as handle:
        status = json.load(handle)
    assert status["selections"]["allow_ckd_download"] is True
    assert status["ready_for_local_run"] is False
    assert status["ckd_sources"]["H2O"]["explicit_table"] is True
    assert status["ckd_sources"]["H2O"]["table_status"] == "not_h5"
    assert status["problems"] == ["ckd_sources.H2O.table: not_h5"]


def test_ckd_check_forward_reports_multiple_local_tables_before_traceback(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O h5 table is ambiguous" in result.stderr
    assert "(2 non-empty files; specify a directory with one table)" in result.stderr
    assert "Traceback" not in result.stderr


def test_ckd_check_forward_reports_multiple_tables_even_when_download_allowed(
    tmp_path,
):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--allow-ckd-download",
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O h5 table is ambiguous" in result.stderr
    assert "(2 non-empty files; specify a directory with one table)" in result.stderr
    assert "Traceback" not in result.stderr


def test_ckd_table_path_override_selects_one_table_from_multiple(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_path = write_synthetic_exomolop_ckd(ckd_root)
    table_dir = ckd_root / "H2O" / "1H2-16O" / "POKAZATEL"
    (table_dir / "another.h5").write_bytes(b"not a real h5 table")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--ckd-table-paths",
        f"H2O={table_path}",
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 0, result.stderr
    assert "Forward-model check:" in result.stdout
    assert "  finite model: True" in result.stdout


def test_ckd_check_forward_reports_non_h5_explicit_table_before_traceback(tmp_path):
    ckd_root = tmp_path / "ckd"
    table_path = tmp_path / "table.txt"
    table_path.write_text("not an h5 table")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "none",
        "--ckd-root",
        str(ckd_root),
        "--ckd-table-paths",
        f"H2O={table_path}",
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O h5 table is not .h5" in result.stderr
    assert "Traceback" not in result.stderr


def test_ckd_check_forward_reports_invalid_schema_before_traceback(tmp_path):
    ckd_root = tmp_path / "ckd"
    write_synthetic_exomolop_ckd(ckd_root, weights=np.array([0.5, 0.5, 0.5]))
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--max-observed",
        "64",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O h5 schema is invalid_values" in result.stderr
    assert "weights must sum to one" in result.stderr
    assert "Traceback" not in result.stderr


def test_ckd_check_forward_reports_missing_local_table_before_traceback(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")

    ckd_root = tmp_path / "ckd"
    (ckd_root / "H2O" / "1H2-16O" / "POKAZATEL").mkdir(parents=True)

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O h5 table" in result.stderr
    assert "Use --allow-ckd-download to permit CKD downloads" in result.stderr
    assert "Traceback" not in result.stderr


def test_ckd_check_forward_reports_missing_ckd_directory_before_traceback(tmp_path):
    opacity_root = create_minimal_opacity_root(tmp_path / "opacity")
    ckd_root = tmp_path / "ckd"

    result = run_example(
        "--check-forward",
        "--data-mode",
        "wide",
        "--opacity-mode",
        "ckd",
        "--channels",
        "nirspec_g395h",
        "--molecules",
        "H2O",
        "--cia-pairs",
        "H2H2",
        "--opacity-root",
        str(opacity_root),
        "--ckd-root",
        str(ckd_root),
        "--jax-platform",
        "cpu",
        "--rv-min",
        "-100",
        "--rv-max",
        "50",
        "--skip-corner",
    )

    assert result.returncode == 1
    assert "Required local inputs are missing" in result.stderr
    assert "ExoMolOP CKD H2O:" in result.stderr
    assert str(ckd_root / "H2O" / "1H2-16O" / "POKAZATEL") in result.stderr
    assert "Use --allow-ckd-download to permit CKD downloads" in result.stderr
    assert "Traceback" not in result.stderr
