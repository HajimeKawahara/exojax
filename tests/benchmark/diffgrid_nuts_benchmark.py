"""Benchmark PreMODIT and DiffGrid HMC-NUTS retrievals.

The benchmark is split into preparation, method-specific runs, and summary
generation.  Run each opacity method in a fresh process so that JAX allocator
state and process-lifetime device-memory peaks do not leak across methods.

This is a manual GPU benchmark and is not intended for the unit-test suite.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

import exojax
from exojax.database import molinfo
from exojax.database.cia.api import CdbCIA
from exojax.opacity import OpaCIA, OpaDiffgrid, OpaPremodit, saveopa
from exojax.postproc.response import ipgauss_sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.rt import ArtEmisPure
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.grids import velocity_grid, wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("tests/benchmark/output_diffgrid_nuts")
DEFAULT_MDB_PATH = Path(".database/CH4/12C-1H4/YT10to10")
DEFAULT_CIA_PATH = Path(".database/H2-H2_2011.cia")

TRUTH = {
    "radius": 0.88,
    "radial_velocity": 10.0,
    "methane_mass_mixing_ratio": 0.0059,
    "temperature_at_1bar": 1200.0,
    "temperature_index": 0.1,
    "vsini": 20.0,
}

PRIOR_BOUNDS = {
    "radius": (0.4, 1.2),
    "radial_velocity": (5.0, 15.0),
    "methane_mass_mixing_ratio": (0.0, 0.015),
    "temperature_at_1bar": (1000.0, 1500.0),
    "temperature_index": (0.05, 0.2),
    "vsini": (15.0, 25.0),
}


@dataclass(frozen=True)
class CaseConfig:
    """Physical and numerical configuration shared by both methods."""

    observed_wavelength_min: float = 16370.0
    observed_wavelength_max: float = 16550.0
    number_of_observed_wavenumbers: int = 1500
    model_wavelength_margin: float = 10.0
    number_of_wavenumbers: int = 7500
    number_of_layers: int = 100
    pressure_top: float = 1.0e-8
    pressure_bottom: float = 1.0e2
    temperature_min: float = 400.0
    temperature_max: float = 1500.0
    number_of_temperature_nodes: int = 21
    planet_mass: float = 33.2
    instrument_resolution: float = 100000.0
    maximum_vsini: float = 100.0
    noise_sigma: float = 0.05
    flux_scale: float = 20000.0
    mean_molecular_weight: float = 2.33
    hydrogen_mass_mixing_ratio: float = 0.74
    premodit_diffmode: int = 1
    broadening_resolution: float = 0.2
    observation_seed: int = 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return result


def _device_memory_stats() -> dict[str, Any]:
    """Return portable diagnostic memory fields when the backend provides them."""
    device = jax.devices()[0]
    try:
        stats = device.memory_stats()
    except Exception as error:  # pragma: no cover - backend dependent
        return {"available": False, "error": str(error)}
    if stats is None:
        return {"available": False}

    result: dict[str, Any] = {"available": True}
    for key in (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "pool_bytes",
        "peak_pool_bytes",
        "largest_free_block_bytes",
    ):
        value = stats.get(key)
        if value is not None and int(value) >= 0:
            result[key] = int(value)
    return result


def _host_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def _environment() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "exojax": getattr(exojax, "__version__", "unknown"),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpyro": numpyro.__version__,
        "jax_enable_x64": bool(config.values["jax_enable_x64"]),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "device_platform": device.platform,
    }


def _case_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "case": output_dir / "case.npz",
        "prepare": output_dir / "prepare.json",
        "premodit": output_dir / "premodit.npz",
        "premodit_metadata": output_dir / "premodit_metadata.json",
        "diffgrid": output_dir / "diffgrid.npz",
        "diffgrid_metadata": output_dir / "diffgrid_metadata.json",
    }


def _block_opacity(opacity: Any) -> None:
    arrays = []
    for name in (
        "log_cross_section_grid",
        "log_cross_section_derivative_grid",
        "pressure_grid",
        "temperature_grid",
        "lbd_coeff",
        "lbd_coeff_reshaped",
    ):
        value = getattr(opacity, name, None)
        if isinstance(value, jax.Array):
            arrays.append(value)
    if arrays:
        jax.block_until_ready(tuple(arrays))


def _observation_grid(case_config: CaseConfig) -> tuple[np.ndarray, np.ndarray]:
    nu_data, wavelength_data, _ = wavenumber_grid(
        case_config.observed_wavelength_min,
        case_config.observed_wavelength_max,
        case_config.number_of_observed_wavenumbers,
        unit="AA",
        xsmode="modit",
        wavelength_order="ascending",
    )
    return np.asarray(nu_data), np.asarray(wavelength_data)


def _model_grid(
    wavelength_data: np.ndarray, case_config: CaseConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    nu_grid, wavelength_grid, resolution = wavenumber_grid(
        float(np.min(wavelength_data) - case_config.model_wavelength_margin),
        float(np.max(wavelength_data) + case_config.model_wavelength_margin),
        case_config.number_of_wavenumbers,
        unit="AA",
        xsmode="diffgrid",
        wavelength_order="ascending",
    )
    return (
        np.asarray(nu_grid),
        np.asarray(wavelength_grid),
        float(resolution),
    )


def _forward_context(
    nu_grid: np.ndarray,
    nu_data: np.ndarray,
    resolution: float,
    case_config: CaseConfig,
    cia_path: Path,
) -> dict[str, Any]:
    art = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=case_config.pressure_top,
        pressure_btm=case_config.pressure_bottom,
        nlayer=case_config.number_of_layers,
    )
    art.change_temperature_range(
        case_config.temperature_min, case_config.temperature_max
    )

    cia_database = CdbCIA(str(cia_path), nurange=nu_grid)
    opa_cia = OpaCIA(cdb=cia_database, nu_grid=nu_grid)
    hydrogen_molecular_mass = molinfo.molmass_isotope("H2")
    hydrogen_volume_mixing_ratio = (
        case_config.hydrogen_mass_mixing_ratio
        * case_config.mean_molecular_weight
        / hydrogen_molecular_mass
    )
    velocity_array = velocity_grid(resolution, case_config.maximum_vsini)
    instrument_beta = resolution_to_gaussian_std(case_config.instrument_resolution)
    jax.block_until_ready(
        (
            art.pressure,
            cia_database.logac,
            cia_database.tcia,
            cia_database.nucia,
            velocity_array,
        )
    )
    return {
        "art": art,
        "opa_cia": opa_cia,
        "nu_data": jnp.asarray(nu_data),
        "nu_grid": jnp.asarray(nu_grid),
        "velocity_array": velocity_array,
        "instrument_beta": instrument_beta,
        "hydrogen_volume_mixing_ratio": hydrogen_volume_mixing_ratio,
    }


def _make_forward_model(opacity: Any, context: dict[str, Any], case_config: CaseConfig):
    art = context["art"]
    opa_cia = context["opa_cia"]
    nu_data = context["nu_data"]
    nu_grid = context["nu_grid"]
    velocity_array = context["velocity_array"]
    instrument_beta = context["instrument_beta"]
    hydrogen_volume_mixing_ratio = context["hydrogen_volume_mixing_ratio"]

    def forward_model(
        temperature,
        methane_mass_mixing_ratio,
        radius,
        radial_velocity,
        vsini,
    ):
        gravity = gravity_jupiter(Rp=radius, Mp=case_config.planet_mass)
        if opacity.method == "diffgrid":
            cross_section = opacity.xsmatrix(temperature)
        else:
            cross_section = opacity.xsmatrix(temperature, art.pressure)

        methane_profile = art.constant_mmr_profile(methane_mass_mixing_ratio)
        optical_depth_methane = art.opacity_profile_xs(
            cross_section,
            methane_profile,
            opacity.molmass,
            gravity,
        )
        log_cia = opa_cia.logacia_matrix(temperature)
        optical_depth_cia = art.opacity_profile_cia(
            log_cia,
            temperature,
            hydrogen_volume_mixing_ratio,
            hydrogen_volume_mixing_ratio,
            case_config.mean_molecular_weight,
            gravity,
        )
        raw_flux = (
            art.run(optical_depth_methane + optical_depth_cia, temperature)
            / case_config.flux_scale
        )
        rotational_flux = convolve_rigid_rotation(
            raw_flux,
            velocity_array,
            vsini,
            u1=0.0,
            u2=0.0,
        )
        return ipgauss_sampling(
            nu_data,
            nu_grid,
            rotational_flux,
            instrument_beta,
            radial_velocity,
            velocity_array,
        )

    return forward_model


def _make_numpyro_model(
    forward_model,
    art: ArtEmisPure,
    case_config: CaseConfig,
    prior_bounds: dict[str, tuple[float, float]],
):
    def model(observation=None):
        radius = numpyro.sample("radius", dist.Uniform(*prior_bounds["radius"]))
        radial_velocity = numpyro.sample(
            "radial_velocity", dist.Uniform(*prior_bounds["radial_velocity"])
        )
        methane_mass_mixing_ratio = numpyro.sample(
            "methane_mass_mixing_ratio",
            dist.Uniform(*prior_bounds["methane_mass_mixing_ratio"]),
        )
        temperature_at_1bar = numpyro.sample(
            "temperature_at_1bar",
            dist.Uniform(*prior_bounds["temperature_at_1bar"]),
        )
        temperature_index = numpyro.sample(
            "temperature_index",
            dist.Uniform(*prior_bounds["temperature_index"]),
        )
        vsini = numpyro.sample("vsini", dist.Uniform(*prior_bounds["vsini"]))

        temperature = art.powerlaw_temperature(temperature_at_1bar, temperature_index)
        prediction = forward_model(
            temperature,
            methane_mass_mixing_ratio,
            radius,
            radial_velocity,
            vsini,
        )
        numpyro.sample(
            "spectrum",
            dist.Normal(prediction, case_config.noise_sigma),
            obs=observation,
        )

    return model


def _validation_profiles(
    truth: dict[str, float], prior_bounds: dict[str, tuple[float, float]]
):
    temperature_bounds = prior_bounds["temperature_at_1bar"]
    index_bounds = prior_bounds["temperature_index"]
    profiles = [
        (
            "mock parameters",
            truth["temperature_at_1bar"],
            truth["temperature_index"],
        )
    ]
    profiles.extend(
        (f"prior corner {temperature:.0f} K, {index:.2f}", temperature, index)
        for temperature in temperature_bounds
        for index in index_bounds
    )
    return profiles


def prepare(args: argparse.Namespace) -> None:
    from exojax.database.exomol.api import MdbExomol

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _case_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Preparation artifacts already exist: {names}. "
            "Use a new output directory or pass --overwrite."
        )

    case_config = CaseConfig(
        number_of_observed_wavenumbers=args.number_of_observed_wavenumbers,
        number_of_wavenumbers=args.number_of_wavenumbers,
        number_of_layers=args.number_of_layers,
        number_of_temperature_nodes=args.number_of_temperature_nodes,
    )
    mdb_path = args.mdb_path.expanduser().resolve()
    cia_path = args.cia_path.expanduser().resolve()
    device_snapshots = {"process_start": _device_memory_stats()}
    timings: dict[str, float] = {}

    nu_data, wavelength_data = _observation_grid(case_config)
    nu_grid, wavelength_grid, resolution = _model_grid(wavelength_data, case_config)
    art_for_grid = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=case_config.pressure_top,
        pressure_btm=case_config.pressure_bottom,
        nlayer=case_config.number_of_layers,
    )
    art_for_grid.change_temperature_range(
        case_config.temperature_min, case_config.temperature_max
    )

    start = time.perf_counter()
    mdb = MdbExomol(str(mdb_path), nurange=nu_grid, gpu_transfer=False)
    number_of_lines = int(len(mdb.nu_lines))
    timings["database_load_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    teacher = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=case_config.premodit_diffmode,
        auto_trange=(
            case_config.temperature_min,
            case_config.temperature_max,
        ),
        broadening_resolution={
            "mode": "manual",
            "value": case_config.broadening_resolution,
        },
        wavelength_order="ascending",
    )
    _block_opacity(teacher)
    timings["premodit_build_seconds"] = time.perf_counter() - start
    device_snapshots["after_premodit_build"] = _device_memory_stats()

    inverse_temperature_nodes = np.linspace(
        1.0 / case_config.temperature_max,
        1.0 / case_config.temperature_min,
        case_config.number_of_temperature_nodes,
    )
    temperature_nodes = 1.0 / inverse_temperature_nodes
    start = time.perf_counter()
    diffgrid = OpaDiffgrid(
        teacher,
        temperature_grid=temperature_nodes,
        pressure_grid=np.asarray(art_for_grid.pressure),
    )
    _block_opacity(diffgrid)
    timings["diffgrid_build_seconds"] = time.perf_counter() - start
    device_snapshots["after_diffgrid_build"] = _device_memory_stats()

    diffgrid.check_pressure_grid(np.asarray(art_for_grid.pressure))
    context = _forward_context(nu_grid, nu_data, resolution, case_config, cia_path)
    teacher_forward = _make_forward_model(teacher, context, case_config)
    diffgrid_forward = _make_forward_model(diffgrid, context, case_config)
    art = context["art"]

    validation_error_in_noise: dict[str, float] = {}
    teacher_flux = None
    diffgrid_flux = None
    start = time.perf_counter()
    for label, temperature_at_1bar, temperature_index in _validation_profiles(
        TRUTH, PRIOR_BOUNDS
    ):
        temperature = art.powerlaw_temperature(temperature_at_1bar, temperature_index)
        model_arguments = (
            temperature,
            TRUTH["methane_mass_mixing_ratio"],
            TRUTH["radius"],
            TRUTH["radial_velocity"],
            TRUTH["vsini"],
        )
        candidate_teacher_flux = teacher_forward(*model_arguments)
        candidate_diffgrid_flux = diffgrid_forward(*model_arguments)
        jax.block_until_ready((candidate_teacher_flux, candidate_diffgrid_flux))
        validation_error_in_noise[label] = float(
            jnp.max(jnp.abs(candidate_diffgrid_flux - candidate_teacher_flux))
            / case_config.noise_sigma
        )
        if label == "mock parameters":
            teacher_flux = np.asarray(candidate_teacher_flux)
            diffgrid_flux = np.asarray(candidate_diffgrid_flux)
    timings["accuracy_validation_seconds"] = time.perf_counter() - start
    device_snapshots["after_accuracy_validation"] = _device_memory_stats()

    maximum_error = max(validation_error_in_noise.values())
    if maximum_error > args.max_interpolation_error_in_noise:
        raise RuntimeError(
            "DiffGrid interpolation error exceeds the configured limit: "
            f"{maximum_error:.6g} > "
            f"{args.max_interpolation_error_in_noise:.6g}. Increase "
            "--number-of-temperature-nodes."
        )
    assert teacher_flux is not None and diffgrid_flux is not None
    observation_rng = np.random.default_rng(case_config.observation_seed)
    observed_flux = teacher_flux + observation_rng.normal(
        0.0, case_config.noise_sigma, nu_data.size
    )

    start = time.perf_counter()
    saveopa(
        teacher,
        str(paths["premodit"]),
        format="npz",
        extra_meta={"benchmark": "diffgrid_nuts", "method": "premodit"},
    )
    timings["premodit_save_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    saveopa(
        diffgrid,
        str(paths["diffgrid"]),
        format="npz",
        extra_meta={"benchmark": "diffgrid_nuts", "method": "diffgrid"},
    )
    timings["diffgrid_save_seconds"] = time.perf_counter() - start

    np.savez_compressed(
        paths["case"],
        nu_data=nu_data,
        wavelength_data=wavelength_data,
        nu_grid=nu_grid,
        wavelength_grid=wavelength_grid,
        model_resolution=np.asarray(resolution),
        observed_flux=observed_flux,
        teacher_flux=teacher_flux,
        diffgrid_flux=diffgrid_flux,
    )
    table_payload_bytes = int(
        diffgrid.log_cross_section_grid.size
        * diffgrid.log_cross_section_grid.dtype.itemsize
        + diffgrid.log_cross_section_derivative_grid.size
        * diffgrid.log_cross_section_derivative_grid.dtype.itemsize
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(),
        "config": asdict(case_config),
        "truth": TRUTH,
        "prior_bounds": PRIOR_BOUNDS,
        "inputs": {
            "mdb_path": str(mdb_path),
            "cia_path": str(cia_path),
            "cia_sha256": _sha256(cia_path),
            "number_of_lines": number_of_lines,
        },
        "artifacts": {
            "case": str(paths["case"]),
            "case_sha256": _sha256(paths["case"]),
            "premodit": str(paths["premodit"]),
            "premodit_sha256": _sha256(paths["premodit"]),
            "premodit_metadata_sha256": _sha256(paths["premodit_metadata"]),
            "diffgrid": str(paths["diffgrid"]),
            "diffgrid_sha256": _sha256(paths["diffgrid"]),
            "diffgrid_metadata_sha256": _sha256(paths["diffgrid_metadata"]),
        },
        "timings": timings,
        "diffgrid": {
            "table_shape": list(diffgrid.log_cross_section_grid.shape),
            "table_payload_bytes": table_payload_bytes,
            "maximum_interpolation_error_in_noise": maximum_error,
            "interpolation_error_in_noise": validation_error_in_noise,
        },
        "device_memory": device_snapshots,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }
    _write_json(paths["prepare"], payload)
    print(f"Prepared benchmark artifacts in {output_dir}")
    print(f"CH4 lines: {number_of_lines}")
    print(f"DiffGrid build: {timings['diffgrid_build_seconds']:.3f} s")
    print(f"DiffGrid payload: {table_payload_bytes / 2**20:.3f} MiB")
    print(f"Maximum interpolation error/noise: {maximum_error:.6g}")


def _load_case(output_dir: Path):
    paths = _case_paths(output_dir)
    if not paths["prepare"].exists() or not paths["case"].exists():
        raise FileNotFoundError(
            f"Preparation artifacts are missing in {output_dir}. Run prepare first."
        )
    metadata = _read_json(paths["prepare"])
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark artifact schema version.")
    expected_digest = metadata["artifacts"]["case_sha256"]
    actual_digest = _sha256(paths["case"])
    if actual_digest != expected_digest:
        raise ValueError("case.npz digest does not match prepare.json.")
    case = np.load(paths["case"], allow_pickle=False)
    return paths, metadata, case, actual_digest


def _minimum_effective_sample_size(samples: dict[str, np.ndarray]) -> float | None:
    try:
        diagnostics = summary(samples, group_by_chain=True)
    except Exception:
        return None
    values = [
        float(site["n_eff"])
        for site in diagnostics.values()
        if np.isfinite(site["n_eff"])
    ]
    return min(values) if values else None


def _compiled_memory_analysis(compiled: Any) -> dict[str, int] | None:
    """Return stable byte fields from JAX compiled memory analysis."""
    try:
        analysis = compiled.memory_analysis()
    except Exception:  # pragma: no cover - backend dependent
        return None
    if analysis is None:
        return None
    result = {}
    for field in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    ):
        value = getattr(analysis, field, None)
        if value is not None and int(value) >= 0:
            result[field] = int(value)
    if not result:
        return None
    required = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    if all(field in result for field in required):
        result["total_size_in_bytes"] = (
            result["argument_size_in_bytes"]
            + result["output_size_in_bytes"]
            + result["temp_size_in_bytes"]
            - result["alias_size_in_bytes"]
        )
    return result


def _benchmark_potential_gradient(
    model,
    observation,
    truth: dict[str, float],
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    """Benchmark one compiled reverse-mode potential-and-gradient evaluation."""
    model_info = initialize_model(
        jax.random.PRNGKey(seed),
        model,
        init_strategy=init_to_value(values=truth),
        model_kwargs={"observation": observation},
        forward_mode_differentiation=False,
        validate_grad=False,
    )
    initial_position = model_info.param_info.z
    value_and_grad = jax.jit(jax.value_and_grad(model_info.potential_fn))

    start = time.perf_counter()
    compiled = value_and_grad.lower(initial_position).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(initial_position))

    evaluation_seconds = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = compiled(initial_position)
        jax.block_until_ready(result)
        evaluation_seconds.append(time.perf_counter() - start)
    values = np.asarray(evaluation_seconds)
    return {
        "compile_seconds": compile_seconds,
        "repetitions": repetitions,
        "evaluation_seconds": evaluation_seconds,
        "median_evaluation_seconds": float(np.median(values)),
        "minimum_evaluation_seconds": float(np.min(values)),
        "memory_analysis": _compiled_memory_analysis(compiled),
    }


def run_method(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    paths, prepare_metadata, case, case_digest = _load_case(output_dir)
    case_config = CaseConfig(**prepare_metadata["config"])
    truth = {key: float(value) for key, value in prepare_metadata["truth"].items()}
    prior_bounds = {
        key: tuple(float(value) for value in bounds)
        for key, bounds in prepare_metadata["prior_bounds"].items()
    }
    archive_path = paths[args.method]
    metadata_path = paths[f"{args.method}_metadata"]
    for artifact_name, artifact_path in (
        (args.method, archive_path),
        (f"{args.method}_metadata", metadata_path),
    ):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Opacity artifact is missing: {artifact_path}")
        expected_digest = prepare_metadata["artifacts"][f"{artifact_name}_sha256"]
        if _sha256(artifact_path) != expected_digest:
            raise ValueError(
                f"{artifact_path.name} digest does not match prepare.json."
            )

    memory = {"process_start": _device_memory_stats()}
    start = time.perf_counter()
    if args.method == "premodit":
        opacity = OpaPremodit.from_saved_opa(str(archive_path))
    else:
        opacity = OpaDiffgrid.from_saved_opa(str(archive_path))
    _block_opacity(opacity)
    opacity_load_seconds = time.perf_counter() - start
    memory["after_opacity_load"] = _device_memory_stats()

    nu_grid = np.asarray(case["nu_grid"])
    if not np.array_equal(np.asarray(opacity.nu_grid), nu_grid):
        raise ValueError("Saved opacity and benchmark case use different nu grids.")
    cia_path = Path(prepare_metadata["inputs"]["cia_path"])
    if _sha256(cia_path) != prepare_metadata["inputs"]["cia_sha256"]:
        raise ValueError("CIA input digest does not match prepare.json.")
    start = time.perf_counter()
    context = _forward_context(
        nu_grid,
        np.asarray(case["nu_data"]),
        float(case["model_resolution"]),
        case_config,
        cia_path,
    )
    if opacity.method == "diffgrid":
        opacity.check_pressure_grid(np.asarray(context["art"].pressure))
    forward_model = _make_forward_model(opacity, context, case_config)
    model = _make_numpyro_model(
        forward_model, context["art"], case_config, prior_bounds
    )
    observation = jnp.asarray(case["observed_flux"])
    jax.block_until_ready(observation)
    model_setup_seconds = time.perf_counter() - start
    memory["after_model_setup"] = _device_memory_stats()

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=truth),
        dense_mass=True,
        target_accept_prob=0.95,
        max_tree_depth=10,
        forward_mode_differentiation=False,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=1,
        thinning=1,
        progress_bar=False,
    )
    warmup_key, sampling_key = jax.random.split(jax.random.PRNGKey(args.seed))

    start = time.perf_counter()
    mcmc.warmup(warmup_key, observation=observation)
    jax.block_until_ready(mcmc.last_state)
    compile_and_warmup_seconds = time.perf_counter() - start
    memory["after_warmup"] = _device_memory_stats()

    start = time.perf_counter()
    mcmc.run(
        sampling_key,
        observation=observation,
        extra_fields=("num_steps", "accept_prob"),
    )
    samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((samples, extra_fields))
    sampling_seconds = time.perf_counter() - start
    memory["after_sampling"] = _device_memory_stats()

    samples_host = jax.device_get(samples)
    extra_host = jax.device_get(extra_fields)
    total_num_steps = int(np.sum(extra_host["num_steps"]))
    minimum_ess = _minimum_effective_sample_size(samples_host)
    minimum_ess_per_second = (
        minimum_ess / sampling_seconds if minimum_ess is not None else None
    )
    del samples, extra_fields, mcmc, kernel
    jax.clear_caches()
    gc.collect()

    gradient_benchmark = _benchmark_potential_gradient(
        model,
        observation,
        truth,
        args.seed + 1,
        args.gradient_repetitions,
    )
    memory["after_gradient_benchmark"] = _device_memory_stats()
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "case_sha256": case_digest,
        "environment": _environment(),
        "run": {
            "seed": args.seed,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "gradient_repetitions": args.gradient_repetitions,
            "num_chains": 1,
            "dense_mass": True,
            "target_accept_probability": 0.95,
            "max_tree_depth": 10,
            "forward_mode_differentiation": False,
        },
        "timings": {
            "opacity_load_seconds": opacity_load_seconds,
            "model_setup_seconds": model_setup_seconds,
            "compile_and_warmup_seconds": compile_and_warmup_seconds,
            "sampling_compile_and_run_seconds": sampling_seconds,
            "sampling_seconds_per_sample": sampling_seconds / args.num_samples,
            "cold_milliseconds_per_leapfrog_step": (
                1000.0 * sampling_seconds / total_num_steps
                if total_num_steps > 0
                else None
            ),
        },
        "potential_gradient_benchmark": gradient_benchmark,
        "diagnostics": {
            "total_num_steps": total_num_steps,
            "mean_accept_probability": float(np.mean(extra_host["accept_prob"])),
            "number_of_divergences": int(np.sum(extra_host["diverging"])),
            "minimum_effective_sample_size": _finite_or_none(minimum_ess),
            "minimum_effective_sample_size_per_second": _finite_or_none(
                minimum_ess_per_second
            ),
        },
        "device_memory": memory,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }
    result_path = output_dir / f"{args.method}.json"
    _write_json(result_path, result)
    case.close()
    print(f"Wrote {result_path}")
    print(
        f"{args.method}: warmup={compile_and_warmup_seconds:.3f} s, "
        f"sampling={sampling_seconds:.3f} s, steps={total_num_steps}, "
        "median gradient="
        f"{1000.0 * gradient_benchmark['median_evaluation_seconds']:.6g} ms"
    )


def _memory_field(result: dict[str, Any], phase: str, key: str):
    return result["device_memory"].get(phase, {}).get(key)


def _safe_ratio(numerator: float | None, denominator: float | None):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _comparison_payload(
    prepare_metadata: dict[str, Any], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    premodit = results["premodit"]
    diffgrid = results["diffgrid"]
    premodit_sampling = premodit["timings"]["sampling_compile_and_run_seconds"]
    diffgrid_sampling = diffgrid["timings"]["sampling_compile_and_run_seconds"]
    premodit_gradient = premodit["potential_gradient_benchmark"][
        "median_evaluation_seconds"
    ]
    diffgrid_gradient = diffgrid["potential_gradient_benchmark"][
        "median_evaluation_seconds"
    ]
    premodit_peak = _memory_field(premodit, "after_sampling", "peak_bytes_in_use")
    diffgrid_peak = _memory_field(diffgrid, "after_sampling", "peak_bytes_in_use")
    saved_seconds_per_gradient = premodit_gradient - diffgrid_gradient
    diffgrid_build = prepare_metadata["timings"]["diffgrid_build_seconds"]
    return {
        "schema_version": SCHEMA_VERSION,
        "methods": results,
        "diffgrid_build_seconds": diffgrid_build,
        "diffgrid_table_payload_bytes": prepare_metadata["diffgrid"][
            "table_payload_bytes"
        ],
        "maximum_interpolation_error_in_noise": prepare_metadata["diffgrid"][
            "maximum_interpolation_error_in_noise"
        ],
        "sampling_speedup_premodit_over_diffgrid": _safe_ratio(
            premodit_sampling, diffgrid_sampling
        ),
        "potential_gradient_speedup_premodit_over_diffgrid": _safe_ratio(
            premodit_gradient, diffgrid_gradient
        ),
        "peak_device_memory_ratio_premodit_over_diffgrid": _safe_ratio(
            premodit_peak, diffgrid_peak
        ),
        "peak_device_memory_reduction_fraction": (
            1.0 - diffgrid_peak / premodit_peak
            if premodit_peak is not None
            and diffgrid_peak is not None
            and premodit_peak > 0
            else None
        ),
        "diffgrid_break_even_gradient_evaluations": (
            diffgrid_build / saved_seconds_per_gradient
            if saved_seconds_per_gradient > 0.0
            else None
        ),
    }


def _write_comparison_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "opacity_load_seconds",
        "model_setup_seconds",
        "compile_and_warmup_seconds",
        "sampling_compile_and_run_seconds",
        "sampling_seconds_per_sample",
        "total_num_steps",
        "cold_milliseconds_per_leapfrog_step",
        "median_potential_gradient_seconds",
        "minimum_effective_sample_size_per_second",
        "number_of_divergences",
        "mean_accept_probability",
        "pre_first_evaluation_device_bytes",
        "peak_device_bytes",
        "host_peak_rss_bytes",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in ("premodit", "diffgrid"):
            result = results[method]
            writer.writerow(
                {
                    "method": method,
                    **result["timings"],
                    "median_potential_gradient_seconds": result[
                        "potential_gradient_benchmark"
                    ]["median_evaluation_seconds"],
                    "total_num_steps": result["diagnostics"]["total_num_steps"],
                    "minimum_effective_sample_size_per_second": result["diagnostics"][
                        "minimum_effective_sample_size_per_second"
                    ],
                    "number_of_divergences": result["diagnostics"][
                        "number_of_divergences"
                    ],
                    "mean_accept_probability": result["diagnostics"][
                        "mean_accept_probability"
                    ],
                    "pre_first_evaluation_device_bytes": _memory_field(
                        result, "after_model_setup", "bytes_in_use"
                    ),
                    "peak_device_bytes": _memory_field(
                        result, "after_sampling", "peak_bytes_in_use"
                    ),
                    "host_peak_rss_bytes": result["host_peak_rss_bytes"],
                }
            )


def _plot_comparison(
    path: Path,
    prepare_metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("premodit", "diffgrid")
    labels = ("PreMODIT", "DiffGrid")
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    time_fields = (
        ("opacity_load_seconds", "opacity load"),
        ("model_setup_seconds", "model setup"),
        ("compile_and_warmup_seconds", "compile + warmup"),
        ("sampling_compile_and_run_seconds", "sampling (cold scan)"),
    )
    bottom = np.zeros(len(methods))
    for field, label in time_fields:
        values = np.asarray([results[method]["timings"][field] for method in methods])
        axes[0].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Reused-opacity NUTS")
    axes[0].legend(fontsize=8)

    memory_series = (
        ("after_warmup", "peak_bytes_in_use", "warmup peak"),
        ("after_sampling", "peak_bytes_in_use", "NUTS process peak"),
    )
    width = 0.24
    plotted_memory = False
    for index, (phase, field, label) in enumerate(memory_series):
        values = [_memory_field(results[method], phase, field) for method in methods]
        if any(value is not None for value in values):
            plotted_memory = True
            gib = [value / 2**30 if value is not None else np.nan for value in values]
            axes[1].bar(x + (index - 0.5) * width, gib, width, label=label)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("device memory (GiB)")
    axes[1].set_title("Fresh-process device memory")
    if plotted_memory:
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Device memory statistics\nnot available on this backend",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    build_seconds = prepare_metadata["timings"]["diffgrid_build_seconds"]
    error = prepare_metadata["diffgrid"]["maximum_interpolation_error_in_noise"]
    fig.suptitle(
        f"DiffGrid build: {build_seconds:.1f} s; "
        f"max interpolation error/noise: {error:.3g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_results(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    paths = _case_paths(output_dir)
    prepare_metadata = _read_json(paths["prepare"])
    results = {
        method: _read_json(output_dir / f"{method}.json")
        for method in ("premodit", "diffgrid")
    }
    case_digests = {result["case_sha256"] for result in results.values()}
    expected_case_digest = prepare_metadata["artifacts"]["case_sha256"]
    run_settings = {
        json.dumps(result["run"], sort_keys=True) for result in results.values()
    }
    environment_fields = (
        "exojax",
        "jax",
        "jaxlib",
        "numpyro",
        "jax_enable_x64",
        "device",
        "device_kind",
        "device_platform",
    )
    environments = {
        json.dumps(
            {field: result["environment"].get(field) for field in environment_fields},
            sort_keys=True,
        )
        for result in results.values()
    }
    method_labels = {method: result.get("method") for method, result in results.items()}
    schema_versions = {
        prepare_metadata.get("schema_version"),
        *(result.get("schema_version") for result in results.values()),
    }
    if (
        schema_versions != {SCHEMA_VERSION}
        or case_digests != {expected_case_digest}
        or len(run_settings) != 1
        or len(environments) != 1
        or any(method_labels[method] != method for method in results)
    ):
        raise ValueError("Method results do not use the same case and NUTS settings.")
    comparison = _comparison_payload(prepare_metadata, results)
    _write_json(output_dir / "comparison.json", comparison)
    _write_comparison_csv(output_dir / "comparison.csv", results)
    _plot_comparison(output_dir / "comparison.png", prepare_metadata, results)

    print(f"Wrote comparison artifacts in {output_dir}")
    print(
        "Sampling speedup (PreMODIT / DiffGrid): "
        f"{comparison['sampling_speedup_premodit_over_diffgrid']:.4g}"
    )
    print(
        "Potential-gradient speedup (PreMODIT / DiffGrid): "
        f"{comparison['potential_gradient_speedup_premodit_over_diffgrid']:.4g}"
    )
    memory_ratio = comparison["peak_device_memory_ratio_premodit_over_diffgrid"]
    if memory_ratio is not None:
        print(f"Peak device-memory ratio: {memory_ratio:.4g}")
    else:
        print("Peak device-memory ratio: unavailable")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Build and save the common benchmark artifacts."
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--mdb-path", type=Path, default=DEFAULT_MDB_PATH)
    prepare_parser.add_argument("--cia-path", type=Path, default=DEFAULT_CIA_PATH)
    prepare_parser.add_argument(
        "--number-of-observed-wavenumbers", type=int, default=1500
    )
    prepare_parser.add_argument("--number-of-wavenumbers", type=int, default=7500)
    prepare_parser.add_argument("--number-of-layers", type=int, default=100)
    prepare_parser.add_argument("--number-of-temperature-nodes", type=int, default=21)
    prepare_parser.add_argument(
        "--max-interpolation-error-in-noise", type=float, default=0.01
    )
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    run_parser = subparsers.add_parser(
        "run", help="Run one opacity method in the current fresh process."
    )
    run_parser.add_argument("--method", choices=("premodit", "diffgrid"), required=True)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--num-warmup", type=_positive_int, default=500)
    run_parser.add_argument("--num-samples", type=_positive_int, default=1000)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--gradient-repetitions", type=_positive_int, default=5)
    run_parser.set_defaults(handler=run_method)

    summary_parser = subparsers.add_parser(
        "summarize", help="Combine method JSON files and make the comparison plot."
    )
    summary_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    summary_parser.set_defaults(handler=summarize_results)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
