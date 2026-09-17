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
import json
import os
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROCESS_START = time.perf_counter()

from jax import config

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import exojax


from diffgrid_nuts_storage import (
    SCHEMA_VERSION,
    collect_provenance,
    load_samples,
    read_metadata as _read_json,
    reserve_result,
    result_paths,
    save_samples,
    sha256 as _sha256,
    validate_run_id,
    write_json as _write_json,
    write_npz,
)

DEFAULT_OUTPUT_DIR = Path("tests/benchmark/output_diffgrid_nuts")
DEFAULT_MDB_PATH = Path(".database/CH4/12C-1H4/YT10to10")
DEFAULT_CIA_PATH = Path(".database/H2-H2_2011.cia")


def _load_scientific_runtime():
    """Import numerical constants only after the execution entry selects x64."""
    global molinfo, CdbCIA, OpaCIA, OpaDiffgrid, OpaPremodit, saveopa
    global ipgauss_sampling, convolve_rigid_rotation, ArtEmisPure, gravity_jupiter
    global velocity_grid, wavenumber_grid, resolution_to_gaussian_std
    from exojax.database import molinfo
    from exojax.database.cia.api import CdbCIA
    from exojax.opacity import OpaCIA, OpaDiffgrid, OpaPremodit, saveopa
    from exojax.postproc.response import ipgauss_sampling
    from exojax.postproc.spin_rotation import convolve_rigid_rotation
    from exojax.rt import ArtEmisPure
    from exojax.utils.astrofunc import gravity_jupiter
    from exojax.utils.grids import velocity_grid, wavenumber_grid
    from exojax.utils.instfunc import resolution_to_gaussian_std


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
    # Missing fields in existing prepared cases retain the original teacher.
    profile_kernel: str = "analytic"


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
    try:
        numpyro_version = version("numpyro")
    except PackageNotFoundError:
        numpyro_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "exojax": getattr(exojax, "__version__", "unknown"),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpyro": numpyro_version,
        "numpy": np.__version__,
        "arviz": _optional_version("arviz"),
        "jax_enable_x64": bool(config.values["jax_enable_x64"]),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "device_platform": device.platform,
        "jax_runtime": {
            key: value
            for key, value in config.values.items()
            if "cache" in key and isinstance(value, (str, int, float, bool, type(None)))
        },
    }


def _optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


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
        "cia_temperature_grid": np.asarray(cia_database.tcia),
        "cia_wavenumber_grid": np.asarray(cia_database.nucia),
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
    import numpyro
    import numpyro.distributions as dist

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


@contextmanager
def _record_execution(path: Path, state: dict[str, Any]):
    """A killed process leaves partial metadata; handled failures retain a cause."""
    state.update(schema_version=SCHEMA_VERSION, status="partial", stage="initializing")
    _write_json(path, state)
    try:
        yield state
    except BaseException as error:
        state.update(
            status="failed",
            failure={
                "stage": state["stage"],
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        _write_json(path, state)
        raise
    else:
        state.update(status="completed", stage="completed")
        _write_json(path, state)


def _stage(state, path, name):
    state["stage"] = name
    _write_json(path, state)


def _provenance(args, inputs):
    script = Path(__file__).resolve()
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "handler"
    }
    return collect_provenance(
        script.parents[2],
        [
            script,
            script.with_name("diffgrid_nuts_storage.py"),
            script.with_name("diffgrid_nuts_validation.py"),
            script.with_name("benchmark_metrics.py"),
            script.with_name("benchmark_inference.py"),
            *inputs,
        ],
        settings,
    )


def _database_provenance(mdb, cia_path):
    """Hash selected source/cache files once, during preparation only."""
    manager = mdb.get_datafile_manager()
    candidates = {}
    for path in mdb.trans_file:
        candidates[Path(path)] = "transition source"
        candidates[Path(manager.cache_file(path))] = "loaded transition cache"
    for attribute in ("states_file", "pf_file", "def_file", "broad_file"):
        value = getattr(mdb, attribute, None)
        if value is not None:
            candidates[Path(value)] = attribute
            if attribute == "states_file":
                candidates[Path(manager.cache_file(value))] = "states cache"
    files = []
    for path, role in sorted(candidates.items()):
        present = path.is_file()
        files.append(
            {
                "path": str(path.resolve()),
                "role": role,
                "sha256": _sha256(path) if present else None,
                "reason": None
                if present
                else "File unavailable; the adapter may use a cache.",
            }
        )
    return {
        "molecular_database": {
            "provider": "ExoMol",
            "dataset": mdb.database,
            "isotopologue": mdb.exact_molecule_name,
            "retrieval_url": None,
            "release": None,
            "reason": "The local adapter does not establish acquisition URL or release.",
            "files": files,
        },
        "cia": {
            "path": str(cia_path),
            "sha256": _sha256(cia_path),
            "provider": None,
            "retrieval_url": None,
            "release": None,
            "reason": "Source and release are not established by the local CIA file.",
        },
    }


def _physical_metadata(context, case_config):
    art = context["art"]
    return {
        "solver": {
            "class": "ArtEmisPure",
            "rtsolver": art.rtsolver,
            "nstream": art.nstream,
        },
        "units": {
            "wavelength": "angstrom",
            "wavenumber": "cm^-1",
            "pressure": "bar",
            "temperature": "K",
            "velocity": "km/s",
            "radius": "Jupiter radius",
            "mass": "Jupiter mass",
            "cross_section": "cm^2/molecule",
            "flux": "F_nu / flux_scale; F_nu in erg/s/cm^2/(cm^-1)",
            "composition": "mass mixing ratio",
        },
        "temperature": {
            "profile": "T0 * pressure**alpha",
            "clip": [case_config.temperature_min, case_config.temperature_max],
        },
        "observation": [
            "CH4 + H2-H2 CIA",
            "pure emission",
            "divide by flux_scale",
            "rigid rotation (u1=u2=0)",
            "Gaussian LSF",
            "RV and sampling",
        ],
        "noise": "independent Normal with fixed noise_sigma",
        "opacity": {
            "premodit": "saved PreMODIT teacher",
            "diffgrid": "saved DiffGrid of the same teacher at fixed pressures",
        },
    }


def prepare(args: argparse.Namespace) -> None:
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

    with _record_execution(paths["prepare"], {}) as state:
        config.update("jax_enable_x64", True)
        _load_scientific_runtime()
        state["provenance"] = _provenance(args, [args.cia_path.expanduser().resolve()])
        _prepare(args, state)


def _prepare(args, state):
    from exojax.database.exomol.api import MdbExomol

    output_dir = args.output_dir.resolve()
    paths = _case_paths(output_dir)
    _stage(state, paths["prepare"], "database_load")

    case_config = CaseConfig(
        number_of_observed_wavenumbers=args.number_of_observed_wavenumbers,
        number_of_wavenumbers=args.number_of_wavenumbers,
        number_of_layers=args.number_of_layers,
        number_of_temperature_nodes=args.number_of_temperature_nodes,
        profile_kernel=args.profile_kernel,
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
    database_provenance = _database_provenance(mdb, cia_path)
    _stage(state, paths["prepare"], "opacity_build")

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
        profile_kernel=case_config.profile_kernel,
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
        profile_kernel=case_config.profile_kernel,
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
    _stage(state, paths["prepare"], "accuracy_validation")
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
    state["validation"] = {
        "maximum_error_in_noise": _finite_or_none(maximum_error),
        "error_in_noise": validation_error_in_noise,
    }
    if (
        not all(np.isfinite(value) for value in validation_error_in_noise.values())
        or maximum_error > args.max_interpolation_error_in_noise
    ):
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
    _stage(state, paths["prepare"], "artifact_save")
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

    write_npz(
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
        "physics": _physical_metadata(context, case_config),
        "database_provenance": database_provenance,
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
    state.update(payload)
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
    expected_digest = metadata["artifacts"]["case_sha256"]
    actual_digest = _sha256(paths["case"])
    if actual_digest != expected_digest:
        raise ValueError("case.npz digest does not match prepare.json.")
    case = np.load(paths["case"], allow_pickle=False)
    return paths, metadata, case, actual_digest


def _minimum_effective_sample_size(samples: dict[str, np.ndarray]) -> float | None:
    try:
        from numpyro.diagnostics import summary

        diagnostics = summary(samples, group_by_chain=True)
    except Exception:
        return None
    values = [
        float(site["n_eff"])
        for site in diagnostics.values()
        if np.isfinite(site["n_eff"])
    ]
    return min(values) if values else None


def _sample_diagnostics(samples, extra_fields, sampling_seconds):
    minimum_ess = _minimum_effective_sample_size(samples)
    return {
        "total_num_steps": int(np.sum(extra_fields["num_steps"])),
        "mean_accept_probability": _finite_or_none(
            np.mean(extra_fields["accept_prob"])
        ),
        "number_of_divergences": int(np.sum(extra_fields["diverging"])),
        "minimum_effective_sample_size": _finite_or_none(minimum_ess),
        "minimum_effective_sample_size_per_second": _finite_or_none(
            minimum_ess / sampling_seconds if minimum_ess is not None else None
        ),
    }


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
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

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
    saved = reserve_result(args.output_dir.resolve(), args.method, args.run_id)
    state = {"method": args.method, "run_id": args.run_id}
    with _record_execution(saved["result"], state):
        config.update("jax_enable_x64", True)
        _load_scientific_runtime()
        state["provenance"] = _provenance(args, [args.output_dir / "prepare.json"])
        _run_method(args, state, saved)
        state["process"] = {
            "pid": os.getpid(),
            "elapsed_seconds": time.perf_counter() - _PROCESS_START,
            "scope": "Benchmark module entry through result construction; excludes interpreter startup, final JSON write, and shutdown. Launcher /usr/bin/time records whole-process wall/user/system time.",
        }
    print(f"Wrote {saved['result']}")


def _chain_initialization(args, truth, prior_bounds):
    """Use identical, recorded physical starts and random keys for both methods."""
    if args.chain_method == "parallel" and args.num_chains > jax.local_device_count():
        raise ValueError("Parallel chains require at least one local device per chain.")
    seed = args.seed if args.initialization_seed is None else args.initialization_seed
    if args.initialization == "prior":
        rng = np.random.default_rng(seed)
        # Restrict initialization only, not the model's prior. Avoid infinite logits.
        unit = rng.uniform(0.05, 0.95, (args.num_chains, len(truth)))
        values = [
            {
                name: float(
                    prior_bounds[name][0]
                    + row[index] * (prior_bounds[name][1] - prior_bounds[name][0])
                )
                for index, name in enumerate(truth)
            }
            for row in unit
        ]
    else:
        values = [dict(truth) for _ in range(args.num_chains)]
    warmup_key, sampling_key = jax.random.split(jax.random.PRNGKey(args.seed))
    if args.num_chains > 1:
        warmup_key = jax.random.split(warmup_key, args.num_chains)
        sampling_key = jax.random.split(sampling_key, args.num_chains)
    metadata = {
        "policy": args.initialization,
        "seed": seed,
        "physical_values": values,
        "warmup_keys": np.asarray(warmup_key).reshape(-1, 2).tolist(),
        "sampling_keys": np.asarray(sampling_key).reshape(-1, 2).tolist(),
        "prior_fraction_range": [0.05, 0.95]
        if args.initialization == "prior"
        else None,
    }
    initial = None
    if args.num_chains > 1 or args.initialization != "truth":
        initial = {}
        for name, (lower, upper) in prior_bounds.items():
            unit = (np.asarray([row[name] for row in values]) - lower) / (upper - lower)
            if np.any(unit <= 0) or np.any(unit >= 1) or not np.all(np.isfinite(unit)):
                raise ValueError(
                    "Initialization must lie strictly inside each Uniform prior."
                )
            q = jnp.asarray(np.log(unit) - np.log1p(-unit))
            initial[name] = q if args.num_chains > 1 else q[0]
    return metadata, initial, warmup_key, sampling_key


def _benchmark_forward(forward, art, truth, repetitions):
    arguments = (
        art.powerlaw_temperature(
            truth["temperature_at_1bar"], truth["temperature_index"]
        ),
        *[
            jnp.asarray(truth[name])
            for name in (
                "methane_mass_mixing_ratio",
                "radius",
                "radial_velocity",
                "vsini",
            )
        ],
    )
    start = time.perf_counter()
    compiled = jax.jit(forward).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(*arguments))
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        durations.append(time.perf_counter() - start)
    return {
        "compile_seconds": compile_seconds,
        "repetitions": repetitions,
        "evaluation_seconds": durations,
        "median_evaluation_seconds": float(np.median(durations)),
        "memory_analysis": _compiled_memory_analysis(compiled),
    }


def _posterior_predictive(
    path, forward, art, samples, nu_data, noise_sigma, count, seed
):
    from benchmark_inference import predictive_summary

    if count == 0:
        return {"available": False, "reason": "Disabled with --predictive-draws 0."}
    nchain, ndraw = next(iter(samples.values())).shape[:2]
    indices = np.linspace(0, ndraw - 1, min(count, ndraw), dtype=int)
    compiled = jax.jit(forward)
    prediction = []
    # Sequential forwards avoid replicating large opacity intermediates per draw.
    for chain in range(nchain):
        values = []
        for draw in indices:
            p = {
                name: jnp.asarray(array[chain, draw]) for name, array in samples.items()
            }
            temperature = art.powerlaw_temperature(
                p["temperature_at_1bar"], p["temperature_index"]
            )
            values.append(
                np.asarray(
                    compiled(
                        temperature,
                        p["methane_mass_mixing_ratio"],
                        p["radius"],
                        p["radial_velocity"],
                        p["vsini"],
                    )
                )
            )
        prediction.append(values)
    prediction = np.asarray(prediction)
    noise_seed = [seed & 0xFFFFFFFF, 707]
    rng = np.random.default_rng(noise_seed)
    replicated = prediction + noise_sigma * rng.normal(size=prediction.shape)
    write_npz(
        path,
        prediction=prediction,
        replicated_observation=replicated,
        draw_indices=indices,
        nu_data=np.asarray(nu_data),
    )
    return {
        "available": True,
        "filename": path.name,
        "sha256": _sha256(path),
        "shape": list(prediction.shape),
        "draw_indices": indices.tolist(),
        "noise_seed": noise_seed,
        "prediction": predictive_summary(prediction),
        "replicated_observation": predictive_summary(replicated, includes_noise=True),
        "scope": "Evenly spaced retained draws in every chain; noiseless prediction and fixed-noise replicated observation. No coverage claim from one mock.",
    }


def _validate_artifacts(paths, metadata, methods):
    for method in methods:
        for name in (method, f"{method}_metadata"):
            if _sha256(paths[name]) != metadata["artifacts"][f"{name}_sha256"]:
                raise ValueError(
                    f"{paths[name].name} digest does not match prepare.json."
                )
    cia_path = Path(metadata["inputs"]["cia_path"])
    if _sha256(cia_path) != metadata["inputs"]["cia_sha256"]:
        raise ValueError("CIA input digest does not match prepare.json.")


def _load_opacity(method, archive_path, allow_code_revision=False):
    """Relax only code version matching for explicit revision reuse.

    DiffGrid loading has two stages because its public ``strict`` option also
    controls device dtype conversion, which must remain strict here.
    """
    opacity_class = {"premodit": OpaPremodit, "diffgrid": OpaDiffgrid}[method]
    if not allow_code_revision:
        return opacity_class.from_saved_opa(str(archive_path))
    if method == "premodit":
        return opacity_class.from_saved_opa(str(archive_path), strict=False)
    from exojax.opacity.diffgrid.io import load_diffgrid_payload

    arrays, metadata = load_diffgrid_payload(str(archive_path), strict=False)
    opacity = OpaDiffgrid.__new__(OpaDiffgrid)
    opacity._init_from_saved_payload(arrays, metadata, strict=True)
    return opacity


def _run_method(args, state, saved):
    from benchmark_inference import DEFAULT_RULES, posterior_diagnostics
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    output_dir = args.output_dir.resolve()

    def auxiliary(name):
        return saved["result"].with_name(
            f"{args.method}_{name}" if args.run_id is None else name
        )

    _stage(state, saved["result"], "input_validation")
    paths, prepare_metadata, case, case_digest = _load_case(output_dir)
    # Materialize and close the archive even when subsequent setup fails.
    with case:
        case = dict(case)
    _validate_artifacts(paths, prepare_metadata, [args.method])
    state.update(case_sha256=case_digest, prepare_sha256=_sha256(paths["prepare"]))
    state["validation"] = _validation_gate(
        output_dir,
        state["prepare_sha256"],
        state["provenance"]["code_sha256"],
        validation_id=args.validation_id,
        environment=_environment(),
    )
    case_config = CaseConfig(**prepare_metadata["config"])
    truth = {key: float(value) for key, value in prepare_metadata["truth"].items()}
    prior_bounds = {
        key: tuple(float(value) for value in bounds)
        for key, bounds in prepare_metadata["prior_bounds"].items()
    }
    initialization, initial_params, warmup_key, sampling_key = _chain_initialization(
        args, truth, prior_bounds
    )
    state["quality_rules"] = dict(DEFAULT_RULES)
    state["run"] = {
        "seed": args.seed,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "gradient_repetitions": args.gradient_repetitions,
        "num_chains": args.num_chains,
        "chain_method": args.chain_method,
        "initialization": initialization,
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.local_devices()],
        "predictive_draws": args.predictive_draws,
        "measure_steady_sampling": args.measure_steady_sampling,
        "dense_mass": True,
        "target_accept_probability": 0.95,
        "max_tree_depth": 10,
        "forward_mode_differentiation": False,
    }
    archive_path = paths[args.method]
    _stage(state, saved["result"], "opacity_load")
    memory = {"process_start": _device_memory_stats()}
    start = time.perf_counter()
    opacity = _load_opacity(args.method, archive_path, args.allow_code_revision)
    _block_opacity(opacity)
    opacity_load_seconds = time.perf_counter() - start
    memory["after_opacity_load"] = _device_memory_stats()

    nu_grid = np.asarray(case["nu_grid"])
    if not np.array_equal(np.asarray(opacity.nu_grid), nu_grid):
        raise ValueError("Saved opacity and benchmark case use different nu grids.")
    cia_path = Path(prepare_metadata["inputs"]["cia_path"])
    _stage(state, saved["result"], "model_setup")
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
    state["physics"] = _physical_metadata(context, case_config)
    state["environment"] = _environment()

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
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        thinning=1,
        progress_bar=False,
    )
    _stage(state, saved["result"], "warmup")
    start = time.perf_counter()
    mcmc.warmup(warmup_key, observation=observation, init_params=initial_params)
    jax.block_until_ready(mcmc.last_state)
    compile_and_warmup_seconds = time.perf_counter() - start
    memory["after_warmup"] = _device_memory_stats()

    _stage(state, saved["result"], "sampling")
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
    _stage(state, saved["result"], "sample_save")
    state["samples"] = save_samples(
        saved["samples"], samples_host, extra_host, list(TRUTH)
    )
    total_num_steps = int(np.sum(extra_host["num_steps"]))
    diagnostics = _sample_diagnostics(samples_host, extra_host, sampling_seconds)
    state["posterior_inference"] = posterior_diagnostics(
        samples_host, extra_host, state["quality_rules"]
    )
    state["steady_sampling"] = {
        "available": False,
        "seconds": None,
        "reason": "Not requested; use --measure-steady-sampling for a separate warm-cache continuation.",
    }
    if args.measure_steady_sampling:
        _stage(state, saved["result"], "steady_sampling")
        mcmc.post_warmup_state = mcmc.last_state
        # Continue from the recorded cold trajectory; retain its RNG state.
        continuation_key = mcmc.last_state.rng_key
        start = time.perf_counter()
        mcmc.run(
            continuation_key,
            observation=observation,
            extra_fields=("num_steps", "accept_prob"),
        )
        steady_samples = mcmc.get_samples(group_by_chain=True)
        steady_extra = mcmc.get_extra_fields(group_by_chain=True)
        jax.block_until_ready((steady_samples, steady_extra))
        seconds = time.perf_counter() - start
        state["steady_samples"] = save_samples(
            auxiliary("steady_samples.npz"),
            jax.device_get(steady_samples),
            jax.device_get(steady_extra),
            list(TRUTH),
        )
        state["steady_sampling"] = {
            "available": True,
            "seconds": seconds,
            "reason": None,
            "initial_rng_keys": np.asarray(continuation_key).reshape(-1, 2).tolist(),
            "scope": "Same-shape warm-cache continuation including dispatch and synchronization; may include wrapper recompilation. Pure steady-state sampling is not isolated. Secondary draws are excluded from the primary diagnostics.",
        }
        memory["after_steady_sampling"] = _device_memory_stats()
        del steady_samples, steady_extra
    del samples, extra_fields, mcmc, kernel
    jax.clear_caches()
    gc.collect()

    _stage(state, saved["result"], "forward_benchmark")
    state["forward_benchmark"] = _benchmark_forward(
        forward_model, context["art"], truth, args.gradient_repetitions
    )
    jax.clear_caches()
    gc.collect()
    _stage(state, saved["result"], "gradient_benchmark")
    gradient_benchmark = _benchmark_potential_gradient(
        model,
        observation,
        truth,
        args.seed + 1,
        args.gradient_repetitions,
    )
    memory["after_gradient_benchmark"] = _device_memory_stats()
    _stage(state, saved["result"], "posterior_predictive")
    start = time.perf_counter()
    state["posterior_predictive"] = _posterior_predictive(
        auxiliary("posterior_predictive.npz"),
        forward_model,
        context["art"],
        samples_host,
        case["nu_data"],
        case_config.noise_sigma,
        args.predictive_draws,
        args.seed,
    )
    predictive_seconds = time.perf_counter() - start
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "case_sha256": case_digest,
        "environment": _environment(),
        "timings": {
            "opacity_load_seconds": opacity_load_seconds,
            "model_setup_seconds": model_setup_seconds,
            "compile_and_warmup_seconds": compile_and_warmup_seconds,
            "sampling_compile_and_run_seconds": sampling_seconds,
            "sampling_seconds_per_sample": sampling_seconds
            / (args.num_chains * args.num_samples),
            "posterior_predictive_seconds": predictive_seconds,
            "cold_milliseconds_per_leapfrog_step": (
                1000.0 * sampling_seconds / total_num_steps
                if total_num_steps > 0
                else None
            ),
        },
        "potential_gradient_benchmark": gradient_benchmark,
        "diagnostics": diagnostics,
        "device_memory": memory,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
        "measurement_definitions": {
            "compile_and_warmup_seconds": "Combined initialization, compilation, and adaptation; synchronized.",
            "sampling_compile_and_run_seconds": "First post-warmup sampling call, including cold-scan compilation and synchronization.",
            "sampling_seconds_per_sample": "Cold sampling seconds divided by chains times retained draws per chain.",
            "host_peak_rss_bytes": "resource.getrusage(RUSAGE_SELF).ru_maxrss, process-lifetime peak, converted to bytes.",
            "device_memory": "First device only: jax.devices()[0].memory_stats snapshots; peak_bytes_in_use is process lifetime, not an isolated phase or sum across parallel chain devices. Missing fields remain unavailable.",
        },
    }
    state.update(result)
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
        "status": "completed",
        "methods": results,
        "device_memory_scope": "First device only; no aggregate peak across parallel chain devices.",
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


def _write_comparison_csv(
    path: Path, results: dict[str, dict[str, Any]], repetitions=None
) -> None:
    fieldnames = [
        "run_id",
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
        "posterior_predictive_seconds",
        "quality_passed",
        "max_rhat_rank",
        "min_ess_bulk",
        "min_ess_tail",
        "warm_cache_sampling_seconds",
        "forward_compile_seconds",
        "median_forward_seconds",
        "process_elapsed_seconds",
        "whole_process_wall_seconds",
        "bulk_ess_per_cold_second",
        "tail_ess_per_cold_second",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in [
            pair[method]
            for pair in (repetitions or [results])
            for method in ("premodit", "diffgrid")
        ]:
            method = result["method"]
            writer.writerow(
                {
                    "method": method,
                    "run_id": result.get("run_id"),
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
                    "quality_passed": result.get("posterior_inference", {}).get(
                        "quality_passed"
                    ),
                    **{
                        key: result.get("posterior_inference", {})
                        .get("summary", {})
                        .get(key)
                        for key in ("max_rhat_rank", "min_ess_bulk", "min_ess_tail")
                    },
                    "warm_cache_sampling_seconds": result.get(
                        "steady_sampling", {}
                    ).get("seconds"),
                    "forward_compile_seconds": result.get("forward_benchmark", {}).get(
                        "compile_seconds"
                    ),
                    "median_forward_seconds": result.get("forward_benchmark", {}).get(
                        "median_evaluation_seconds"
                    ),
                    "process_elapsed_seconds": result.get("process", {}).get(
                        "elapsed_seconds"
                    ),
                    "whole_process_wall_seconds": result.get(
                        "whole_process_time", {}
                    ).get("real"),
                    **{
                        key: result.get("posterior_cost", {}).get(key)
                        for key in (
                            "bulk_ess_per_cold_second",
                            "tail_ess_per_cold_second",
                        )
                    },
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
    axes[1].set_title("First-device process memory")
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


def _validate_results(prepare_metadata, results, revision_comparison=False):
    """Revision comparisons relax code identity only, retaining experimental controls."""
    for result in results:
        schema = result.get("schema_version")
        if schema not in (1, SCHEMA_VERSION):
            raise ValueError(f"Unsupported benchmark result schema version: {schema}")
        if schema == SCHEMA_VERSION and result.get("status") != "completed":
            raise ValueError("Only completed runs can be compared.")
        if result["case_sha256"] != prepare_metadata["artifacts"]["case_sha256"]:
            raise ValueError("Method results do not use the same case.")
    reference = results[0]
    for candidate in results[1:]:
        if candidate["run"] != reference["run"]:
            raise ValueError("Method results do not use the same NUTS settings.")
        if candidate.get("quality_rules") != reference.get("quality_rules"):
            raise ValueError(
                "Method results do not use the same posterior quality rules."
            )
        # Schema 1 lacks Python/platform/cache and full dependency provenance.
        legacy = any(result["schema_version"] == 1 for result in (reference, candidate))
        environment_fields = (
            (
                "exojax",
                "jax",
                "jaxlib",
                "numpyro",
                "jax_enable_x64",
                "device",
                "device_kind",
                "device_platform",
            )
            if legacy
            else set(reference["environment"]) | set(candidate["environment"])
        )
        for field in environment_fields:
            if revision_comparison and field == "exojax":
                continue
            if reference["environment"].get(field) != candidate["environment"].get(
                field
            ):
                raise ValueError(
                    f"Run environment mismatch: {field}: {reference['environment'].get(field)!r} != {candidate['environment'].get(field)!r}"
                )
        if not legacy:
            if reference["physics"] != candidate["physics"]:
                raise ValueError("Run physics mismatch.")
            for field in ("environment", "dependencies"):
                left = dict(reference["provenance"][field])
                right = dict(candidate["provenance"][field])
                if revision_comparison and field == "dependencies":
                    left = {k: v for k, v in left.items() if k.lower() != "exojax"}
                    right = {k: v for k, v in right.items() if k.lower() != "exojax"}
                if left != right:
                    raise ValueError(f"Run provenance mismatch: {field}")
            if not revision_comparison:
                for field in ("code_sha256",):
                    if reference["provenance"][field] != candidate["provenance"][field]:
                        raise ValueError(
                            "Run code differs; select an explicit revision comparison."
                        )
                if (
                    reference["provenance"]["git"]["commit"]
                    != candidate["provenance"]["git"]["commit"]
                ):
                    raise ValueError(
                        "Run git commits differ; select an explicit revision comparison."
                    )


def _read_result(output_dir, method, run_id, prepare_digest):
    from benchmark_inference import posterior_diagnostics, predictive_summary

    paths = result_paths(output_dir, method, run_id)
    result = _read_json(paths["result"])
    if result.get("method") != method or result.get("run_id") != run_id:
        raise ValueError(
            "Result method or run ID does not match the requested selection."
        )
    if result["schema_version"] == SCHEMA_VERSION:
        if result["prepare_sha256"] != prepare_digest:
            raise ValueError(
                "Run prepare.json digest does not match the selected case."
            )
        # Always check raw samples, even when stored diagnostics are reused.
        samples, extra = load_samples(paths["samples"], result["samples"])
        expected_shape = [result["run"]["num_chains"], result["run"]["num_samples"]]
        if result["samples"]["chain_shape"] != expected_shape:
            raise ValueError("Saved chains do not match the run's chain/draw counts.")
        result["diagnostics"] = _sample_diagnostics(
            samples, extra, result["timings"]["sampling_compile_and_run_seconds"]
        )
        result["diagnostics_source"] = "recomputed from saved chains"
        result["posterior_inference"] = posterior_diagnostics(
            samples, extra, result.get("quality_rules")
        )
        result["posterior_inference_source"] = (
            "Recomputed from hash-verified primary chains, using the recorded rules and current recorded diagnostic implementation."
        )
        steady = result.get("steady_samples")
        if steady is not None:
            load_samples(
                paths["result"].with_name(validate_run_id(steady["filename"])), steady
            )
            if steady["chain_shape"] != expected_shape:
                raise ValueError(
                    "Steady sampling chains do not match the run dimensions."
                )
        predictive = result.get("posterior_predictive", {})
        if predictive.get("available"):
            prediction_path = paths["result"].with_name(
                validate_run_id(predictive["filename"])
            )
            if _sha256(prediction_path) != predictive["sha256"]:
                raise ValueError(
                    "Posterior predictive archive digest does not match result metadata."
                )
            with np.load(prediction_path, allow_pickle=False) as archive:
                if (
                    list(archive["prediction"].shape) != predictive["shape"]
                    or archive["draw_indices"].tolist() != predictive["draw_indices"]
                ):
                    raise ValueError(
                        "Posterior predictive shape or selected draws do not match metadata."
                    )
                with np.load(
                    _case_paths(Path(output_dir))["case"], allow_pickle=False
                ) as case:
                    if not np.array_equal(archive["nu_data"], case["nu_data"]):
                        raise ValueError(
                            "Posterior predictive observation grid differs from the saved case."
                        )
                    shape = (
                        result["run"]["num_chains"],
                        len(predictive["draw_indices"]),
                        len(case["nu_data"]),
                    )
                indices = archive["draw_indices"]
                if (
                    indices.ndim != 1
                    or indices.dtype.kind not in "iu"
                    or np.any(indices < 0)
                    or np.any(indices >= result["run"]["num_samples"])
                    or np.any(np.diff(indices) <= 0)
                ):
                    raise ValueError(
                        "Posterior predictive draw indices are invalid for the run."
                    )
                for name in ("prediction", "replicated_observation"):
                    if archive[name].shape != shape:
                        raise ValueError(
                            "Posterior predictive shape does not match the run and observation."
                        )
                    predictive[name] = predictive_summary(
                        archive[name], includes_noise=name == "replicated_observation"
                    )
        try:
            result["diagnostics_numpyro_version"] = version("numpyro")
        except PackageNotFoundError:
            result["diagnostics_numpyro_version"] = None
        if result["diagnostics"]["minimum_effective_sample_size"] is None:
            result["diagnostics_note"] = (
                "ESS unavailable: NumPyro missing or diagnostics undefined."
            )
    summary = result.get("posterior_inference", {}).get("summary", {})
    seconds = result["timings"]["sampling_compile_and_run_seconds"]
    result["posterior_cost"] = {
        "bulk_ess_per_cold_second": _safe_ratio(summary.get("min_ess_bulk"), seconds),
        "tail_ess_per_cold_second": _safe_ratio(summary.get("min_ess_tail"), seconds),
        "scope": "Minimum over parameters; primary retained-chain ESS divided by cold sampling compile-and-run wall time. Interpret only with comparison quality eligibility.",
    }
    result["whole_process_time"] = _read_process_time(output_dir, method, run_id)
    return result


def _read_process_time(output_dir, method, run_id):
    path = Path(output_dir) / "process_times" / f"{run_id}-{method}.txt"
    if run_id is None or not path.is_file():
        return {
            "available": False,
            "real": None,
            "user": None,
            "sys": None,
            "reason": "Whole-process /usr/bin/time log was not recorded by the launcher.",
        }
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in ("real", "user", "sys"):
            values[fields[0]] = float(fields[1])
    if set(values) != {"real", "user", "sys"} or not all(
        np.isfinite(value) and value >= 0 for value in values.values()
    ):
        raise ValueError(f"Invalid whole-process timing log: {path}")
    return {
        "available": True,
        **values,
        "filename": str(path),
        "sha256": _sha256(path),
        "scope": "Launcher /usr/bin/time -p, including interpreter startup and shutdown.",
    }


def _validation_gate(
    output_dir,
    prepare_digest,
    code_sha256=None,
    required=False,
    validation_id=None,
    environment=None,
):
    """Bind explicit successful evidence; never choose the latest validation."""
    root = Path(output_dir) / "validations"
    if validation_id is None:
        reports = [
            path
            for path in root.glob("*/validation.json")
            if json.loads(path.read_text()).get("prepare_sha256") == prepare_digest
        ]
        if reports or required:
            raise ValueError(
                "Select a successful --validation-id explicitly before using this case; validation evidence cannot be ignored."
            )
        return {"status": "not_run", "passed": None}
    path = root / validate_run_id(validation_id) / "validation.json"
    result = _read_json(path)
    if (
        result.get("validation_id") != validation_id
        or result.get("prepare_sha256") != prepare_digest
    ):
        raise ValueError(
            "Validation ID or prepare digest does not match the selected case."
        )
    if result.get("passed") is not True:
        raise ValueError(
            "Validation failed; this case cannot enter a performance comparison."
        )
    if code_sha256 is not None and result["provenance"]["code_sha256"] != code_sha256:
        raise ValueError("Validation execution code does not match the run.")
    if environment is not None:
        for key in (
            "jax",
            "jaxlib",
            "numpyro",
            "numpy",
            "jax_enable_x64",
            "device_platform",
            "device_kind",
        ):
            if result["environment"].get(key) != environment.get(key):
                raise ValueError(
                    f"Validation environment does not match the run: {key}"
                )
    if _sha256(path.with_name("residuals.npz")) != result["residuals"]["sha256"]:
        raise ValueError(
            "Validation residual archive digest does not match the report."
        )
    return {
        "status": "passed",
        "passed": True,
        "validation_id": validation_id,
        "sha256": _sha256(path),
        "reference_convergence": result["reference_convergence"],
    }


def _validate_result_evidence(output_dir, result, prepare_digest, validation_id):
    recorded = result.get("validation", {})
    selected = (
        validation_id if validation_id is not None else recorded.get("validation_id")
    )
    provenance = result.get("provenance") or {}
    evidence = _validation_gate(
        output_dir,
        prepare_digest,
        provenance.get("code_sha256"),
        validation_id=selected,
        environment=result.get("environment"),
    )
    if (
        recorded.get("sha256") is not None
        and evidence.get("sha256") != recorded["sha256"]
    ):
        raise ValueError(
            "Validation report digest differs from the evidence recorded by the run."
        )
    if evidence["passed"] is True and (
        provenance.get("code_sha256") is None or not result.get("environment")
    ):
        raise ValueError(
            "Run lacks code/environment provenance needed to attach validation."
        )
    result["accuracy_validation"] = evidence


def _comparison_quality(results, pair_count=1):
    from benchmark_inference import DEFAULT_RULES

    reasons = []
    if pair_count < 2:
        reasons.append("At least two independent paired runs are required.")
    for result in results:
        label = f"{result.get('run_id')}/{result.get('method')}"
        checks = {
            "The fixed posterior quality rules must be recorded before sampling.": result.get(
                "quality_rules"
            )
            == DEFAULT_RULES,
            "Observation-space validation is missing or failed.": result.get(
                "accuracy_validation", {}
            ).get("passed")
            is True,
            "Posterior convergence criteria are missing or failed.": result.get(
                "posterior_inference", {}
            ).get("quality_passed")
            is True,
            "Dispersed prior-interior initialization is required.": result["run"]
            .get("initialization", {})
            .get("policy")
            == "prior",
            "At least four chains are required.": result["run"].get("num_chains", 0)
            >= 4,
            "At least 500 warmup steps and 1000 retained draws per chain are required.": result[
                "run"
            ].get("num_warmup", 0)
            >= 500
            and result["run"].get("num_samples", 0) >= 1000,
            "Finite posterior predictions are required.": result.get(
                "posterior_predictive", {}
            )
            .get("prediction", {})
            .get("finite")
            is True,
        }
        reasons.extend(
            f"{label}: {reason}" for reason, passed in checks.items() if not passed
        )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "paired_runs": pair_count,
        "scope": "Convergence and teacher-relative accuracy for the recorded case/backend; no repeated-mock coverage or absolute-accuracy claim.",
    }


def _validate_repeat_results(prepare_metadata, pairs):
    """Allow independent seeds/starts across repeats, preserving all other controls."""
    import copy

    reference = pairs[0]["premodit"]
    seeds, initial_seeds, keys = set(), set(), set()
    for pair in pairs:
        _validate_results(prepare_metadata, list(pair.values()))
        candidate = pair["premodit"]
        initialization = candidate["run"].get("initialization")
        if initialization is None:
            raise ValueError(
                "Independent repeats require recorded initialization and random keys."
            )
        for field in ("policy", "prior_fraction_range"):
            if initialization.get(field) != reference["run"]["initialization"].get(
                field
            ):
                raise ValueError(
                    f"Repeated runs use different initialization settings: {field}"
                )
        seed = candidate["run"]["seed"]
        initial_seed = initialization["seed"]
        current_keys = [
            tuple(key)
            for group in ("warmup_keys", "sampling_keys")
            for key in initialization[group]
        ]
        if (
            seed in seeds
            or initial_seed in initial_seeds
            or len(set(current_keys)) != len(current_keys)
            or keys.intersection(current_keys)
        ):
            raise ValueError(
                "Repeated runs must use independent seeds, initializations, and random keys."
            )
        seeds.add(seed)
        initial_seeds.add(initial_seed)
        keys.update(current_keys)
        normalized = copy.deepcopy(candidate)
        normalized["run"]["seed"] = reference["run"]["seed"]
        normalized["run"]["initialization"] = copy.deepcopy(
            reference["run"]["initialization"]
        )
        _validate_results(prepare_metadata, [reference, normalized])
        if candidate.get("quality_rules") != reference.get("quality_rules"):
            raise ValueError("Repeated runs use different posterior quality rules.")


def _write_diagnostic_table(path, results):
    columns = [
        "run_id",
        "method",
        "parameter",
        "mean",
        "mcse_mean",
        "rhat_rank",
        "ess_bulk",
        "ess_tail",
        "quality_passed",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            inference = result.get("posterior_inference", {})
            for name, parameter in inference.get("per_parameter", {}).items():
                writer.writerow(
                    {
                        "run_id": result.get("run_id"),
                        "method": result["method"],
                        "parameter": name,
                        "quality_passed": inference.get("quality_passed"),
                        **{key: parameter.get(key) for key in columns[3:-1]},
                    }
                )


def _posterior_comparison(left, right):
    from benchmark_inference import posterior_difference

    if not all("posterior_inference" in result for result in (left, right)):
        return {
            "available": False,
            "reason": "Raw posterior diagnostics are unavailable for legacy results.",
        }
    comparison = posterior_difference(
        left["posterior_inference"], right["posterior_inference"]
    )
    predictions = [
        result.get("posterior_predictive", {}).get("prediction", {})
        for result in (left, right)
    ]
    if all(prediction.get("finite") for prediction in predictions):
        comparison["prediction_mean_difference"] = (
            np.asarray(predictions[1]["mean"]) - np.asarray(predictions[0]["mean"])
        ).tolist()
        comparison["prediction_quantile_difference"] = {
            key: (
                np.asarray(predictions[1]["quantiles"][key])
                - np.asarray(predictions[0]["quantiles"][key])
            ).tolist()
            for key in predictions[0]["quantiles"]
        }
    return comparison


def summarize_results(args: argparse.Namespace) -> None:
    destination = args.output_dir.resolve()
    if args.run_id is not None:
        destination = destination / "runs" / args.run_id
    try:
        _summarize_results(args)
    except (ValueError, FileNotFoundError) as error:
        destination.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "run_id": args.run_id,
            "repeat_run_ids": args.repeat_run_id,
            "compare_run_id": args.compare_run_id,
            "reason": str(error),
        }
        _write_json(destination / "comparison_failed.json", failure)
        # Generated summaries must not retain a stale successful ranking after
        # a later validation/hash failure for the same explicit selection.
        target = destination / "comparison.json"
        if args.compare_run_id is not None and args.method is not None:
            target = (
                args.output_dir.resolve()
                / "runs"
                / args.compare_run_id
                / args.method
                / f"comparison_from_{args.run_id}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, failure)
        raise


def _summarize_results(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    paths, prepare_metadata, case, _ = _load_case(output_dir)
    case.close()
    revision = args.compare_run_id is not None
    if args.repeat_run_id and (revision or args.run_id is None):
        raise ValueError(
            "--repeat-run-id requires --run-id and cannot be combined with --compare-run-id."
        )
    if len(set([args.run_id, *args.repeat_run_id])) != 1 + len(args.repeat_run_id):
        raise ValueError("Repeated run IDs must be distinct.")
    if revision and (args.run_id is None or args.method is None):
        raise ValueError("--compare-run-id requires --run-id and --method.")
    if args.method is not None and not revision:
        raise ValueError("--method in summarize requires --compare-run-id.")
    methods = [args.method] if revision else ["premodit", "diffgrid"]
    _validate_artifacts(paths, prepare_metadata, methods)
    prepare_digest = _sha256(paths["prepare"])
    if revision:
        baseline = _read_result(output_dir, args.method, args.run_id, prepare_digest)
        candidate = _read_result(
            output_dir, args.method, args.compare_run_id, prepare_digest
        )
        for result in (baseline, candidate):
            _validate_result_evidence(
                output_dir, result, prepare_digest, args.validation_id
            )
        _validate_results(
            prepare_metadata, [baseline, candidate], revision_comparison=True
        )
        comparison = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "comparison_kind": "revision",
            "method": args.method,
            "baseline_run_id": args.run_id,
            "candidate_run_id": args.compare_run_id,
            "baseline": baseline,
            "candidate": candidate,
            "quality": _comparison_quality([baseline, candidate]),
            "posterior_comparison": _posterior_comparison(baseline, candidate),
            "scientific_sampling_speedup_baseline_over_candidate": None,
            "timing_ratio_scope": "Descriptive single-pair revision comparison; repeated revision pairs are not aggregated by this command.",
            "accuracy_validated": all(
                result["accuracy_validation"]["passed"] is True
                for result in (baseline, candidate)
            ),
            "sampling_speedup_baseline_over_candidate": _safe_ratio(
                baseline["timings"]["sampling_compile_and_run_seconds"],
                candidate["timings"]["sampling_compile_and_run_seconds"],
            ),
            "code_versions": {
                label: {
                    "exojax": result["environment"].get("exojax"),
                    "provenance": result.get("provenance"),
                }
                for label, result in (("baseline", baseline), ("candidate", candidate))
            },
        }
        target = output_dir / "runs" / args.compare_run_id / args.method
        destination = target / f"comparison_from_{args.run_id}.json"
        _write_json(destination, comparison)
        _write_diagnostic_table(
            target / f"diagnostics_from_{args.run_id}.csv", [baseline, candidate]
        )
        if not comparison["quality"]["eligible"]:
            print(
                "Single revision pair: timing ratios are descriptive only; see quality reasons."
            )
        print(f"Wrote revision comparison: {destination}")
        return
    results = {
        method: _read_result(output_dir, method, args.run_id, prepare_digest)
        for method in methods
    }
    for result in results.values():
        _validate_result_evidence(
            output_dir, result, prepare_digest, args.validation_id
        )
    pairs = [results]
    for run_id in args.repeat_run_id:
        pair = {
            method: _read_result(output_dir, method, run_id, prepare_digest)
            for method in methods
        }
        for result in pair.values():
            _validate_result_evidence(
                output_dir, result, prepare_digest, args.validation_id
            )
        pairs.append(pair)
    if len(pairs) > 1:
        _validate_repeat_results(prepare_metadata, pairs)
    else:
        _validate_results(prepare_metadata, list(results.values()))
    all_results = [result for pair in pairs for result in pair.values()]
    quality = _comparison_quality(all_results, len(pairs))
    output_dir = (
        output_dir if args.run_id is None else output_dir / "runs" / args.run_id
    )
    print(
        f"Selected run: {args.run_id if args.run_id is not None else 'legacy root results'}"
    )
    comparison = _comparison_payload(prepare_metadata, results)
    comparison.update(
        comparison_kind="methods",
        run_id=args.run_id,
        accuracy_validated=all(
            result["accuracy_validation"]["passed"] is True for result in all_results
        ),
        quality=quality,
        timing_ratio_scope="Raw timing ratios are descriptive. The scientific ratio is available only when every explicitly selected independent pair satisfies the protocol.",
        posterior_comparison=_posterior_comparison(
            results["premodit"], results["diffgrid"]
        ),
        repetitions=[
            {
                "run_id": pair["premodit"].get("run_id"),
                "methods": pair,
                "sampling_speedup_premodit_over_diffgrid": _safe_ratio(
                    pair["premodit"]["timings"]["sampling_compile_and_run_seconds"],
                    pair["diffgrid"]["timings"]["sampling_compile_and_run_seconds"],
                ),
                "posterior_comparison": _posterior_comparison(
                    pair["premodit"], pair["diffgrid"]
                ),
            }
            for pair in pairs
        ],
    )
    comparison["scientific_sampling_speedup_premodit_over_diffgrid"] = (
        float(
            np.median(
                [
                    pair["sampling_speedup_premodit_over_diffgrid"]
                    for pair in comparison["repetitions"]
                ]
            )
        )
        if quality["eligible"]
        else None
    )
    if not quality["eligible"]:
        print(
            "Comparison protocol is not satisfied: timing ratios are descriptive only."
        )
        for reason in quality["reasons"]:
            print(f"  {reason}")
    _write_json(output_dir / "comparison.json", comparison)
    _write_comparison_csv(output_dir / "comparison.csv", results, pairs)
    _write_diagnostic_table(output_dir / "diagnostics.csv", all_results)
    _plot_comparison(output_dir / "comparison.png", prepare_metadata, results)

    print(f"Wrote comparison artifacts in {output_dir}")
    print(
        "Descriptive cold sampling ratio (PreMODIT / DiffGrid): "
        f"{comparison['sampling_speedup_premodit_over_diffgrid']:.4g}"
    )
    print(
        "Descriptive potential-gradient ratio (PreMODIT / DiffGrid): "
        f"{comparison['potential_gradient_speedup_premodit_over_diffgrid']:.4g}"
    )
    memory_ratio = comparison["peak_device_memory_ratio_premodit_over_diffgrid"]
    if memory_ratio is not None:
        print(f"First-device peak memory ratio: {memory_ratio:.4g}")
    else:
        print("Peak device-memory ratio: unavailable")


def _run_id(value):
    try:
        return validate_run_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        "--profile-kernel",
        choices=("analytic", "real_space"),
        default="real_space",
        help="Kernel used by both the saved PreMODIT teacher and DiffGrid construction.",
    )
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
    run_parser.add_argument("--num-chains", type=_positive_int, default=1)
    run_parser.add_argument(
        "--chain-method",
        choices=("sequential", "parallel", "vectorized"),
        default="sequential",
    )
    run_parser.add_argument(
        "--initialization", choices=("truth", "prior"), default="truth"
    )
    run_parser.add_argument(
        "--initialization-seed",
        type=_nonnegative_int,
        help="Seed for common dispersed starts; omitted uses --seed.",
    )
    run_parser.add_argument(
        "--predictive-draws",
        type=_nonnegative_int,
        default=100,
        help="Maximum evenly spaced posterior predictive draws per chain; 0 disables.",
    )
    run_parser.add_argument(
        "--measure-steady-sampling",
        action="store_true",
        help="Measure and save a separate same-sized warm-cache continuation.",
    )
    run_parser.add_argument("--gradient-repetitions", type=_positive_int, default=5)
    run_parser.add_argument(
        "--run-id",
        type=_run_id,
        help="Save under runs/ID/METHOD; existing method runs are rejected.",
    )
    run_parser.add_argument(
        "--allow-code-revision",
        action="store_true",
        help="Reuse opacity from another ExoJAX version; keep schema, hash, and dtype checks.",
    )
    run_parser.add_argument(
        "--validation-id",
        type=_run_id,
        help="Use this successful validation for the same case and code.",
    )
    run_parser.set_defaults(handler=run_method)

    summary_parser = subparsers.add_parser(
        "summarize", help="Combine method JSON files and make the comparison plot."
    )
    summary_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    summary_parser.add_argument(
        "--run-id",
        type=_run_id,
        help="Read this run only; omitted selects legacy root results.",
    )
    summary_parser.add_argument(
        "--compare-run-id",
        type=_run_id,
        help="Compare this candidate revision with the baseline --run-id.",
    )
    summary_parser.add_argument(
        "--repeat-run-id",
        type=_run_id,
        action="append",
        default=[],
        help="Add an independent pair of method runs; repeat for each explicit ID.",
    )
    summary_parser.add_argument(
        "--method",
        choices=("premodit", "diffgrid"),
        help="Method for an explicit revision comparison.",
    )
    summary_parser.add_argument(
        "--validation-id",
        type=_run_id,
        help="Select successful validation explicitly for existing runs.",
    )
    summary_parser.set_defaults(handler=summarize_results)

    validation_parser = subparsers.add_parser(
        "validate",
        help="Check observed spectra, domains, and gradients of a prepared case.",
    )
    validation_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    validation_parser.add_argument("--validation-id", type=_run_id, required=True)
    validation_parser.add_argument("--seed", type=int, default=0)
    validation_parser.add_argument("--num-prior-points", type=_positive_int, default=16)
    validation_parser.add_argument(
        "--max-interpolation-error-in-noise", type=_nonnegative_float, default=0.01
    )
    validation_parser.add_argument("--max-q", type=_nonnegative_float, default=0.1)
    validation_parser.add_argument(
        "--gradient-tolerance", type=_nonnegative_float, default=1e-3
    )
    validation_parser.add_argument("--allow-code-revision", action="store_true")
    validation_parser.add_argument("--reference-output-dir", type=Path)
    validation_parser.set_defaults(handler=validate)
    return parser


def _nonnegative_float(value):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def _nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def validate(args):
    from diffgrid_nuts_validation import validate_case

    validate_case(args, sys.modules[__name__])


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
