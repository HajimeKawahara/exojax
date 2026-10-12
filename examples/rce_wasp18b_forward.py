"""Run or validate the P7 WASP-18b forward model from a pinned local manifest.

Example: python examples/rce_wasp18b_forward.py --manifest assets.json
    --observation /path/to/emission_candidate.ecsv --output forward.json

Use independent manifests for spectral/pressure refinements and --compare for
their observation-space differences. No retrieval or data download is performed.
"""

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import (
    band_planck_flux, bottom_leakage_fraction, integrate_broadband_fluxes,
)
from _rce_observation import load_emission_candidate, pixel_flux_ratio
from _rce_wasp18b import ColumnSettings, SIGMA_SB, Wasp18bColumn, spectral_error
from _rce_wasp18b_assets import (
    load_ckd_opacities, load_lbl_opacities, load_manifest, load_stellar_spectrum,
)
from exojax.database.cia.api import CdbCIA
from exojax.postproc.ckd import sample_ckd_bands_at_wavelengths


def prepare_column(manifest, observation, *, stream_resolved=False):
    """Require explicit isotope, broadening, gravity and omission assumptions."""
    config = manifest.config
    opacity = config["opacity"]
    mapping = opacity["mapping"]
    if set(mapping) != set(opacity["ckd"]) or set(mapping) != set(opacity["lbl"]):
        raise ValueError("Every absorber needs matching CKD, resolved opacity, and mapping entries.")
    for entry in mapping.values():
        if not entry.get("broadening") or not entry.get("isotope_convention"):
            raise ValueError("Record broadening and isotope_convention for each absorber.")
    for field in ("omitted_absorbers", "scattering", "broadening_limitations"):
        if field not in opacity:
            raise ValueError(f"Record opacity.{field} explicitly.")
    stellar_scattering = config["column"]["settings"].get("stellar_scattering", False)
    scattering = "stellar_two_stream" if stellar_scattering else "omitted"
    if opacity["scattering"] != scattering:
        raise ValueError(f"Column stellar_scattering setting requires opacity.scattering='{scattering}'.")
    if not config["column"].get("gravity_source"):
        raise ValueError("column.gravity_source must identify the planetary gravity source.")
    settings = ColumnSettings(**config["column"]["settings"])
    cia = {}
    for name, pair in (("H2-H2", ("H2", "H2")), ("H2-He", ("H2", "He1"))):
        spec = config["cia"][name]
        if spec.get("outside_spectral_range") != "zero":
            raise ValueError("CIA spectral truncation must be explicitly declared as zero.")
        cia[pair] = CdbCIA(str(manifest.paths[spec["asset"]]))
    resolved_stream = None
    if stream_resolved:
        from _rce_wasp18b_streaming import load_streaming_opacities

        lbl, resolved_stream = load_streaming_opacities(manifest)
    else:
        lbl = load_lbl_opacities(manifest)
    column = config["column"]
    prepared = Wasp18bColumn(
        observation, load_stellar_spectrum(manifest),
        load_ckd_opacities(manifest), lbl, cia,
        column["pressure_boundaries_bar"], settings,
        isotope_fractions={name: entry["isotope_fraction"] for name, entry in mapping.items()},
        temperature_initial=column["temperature_initial_K"],
        bottom_temperature_initial=column["bottom_temperature_initial_K"],
        solver_options=column.get("solver_options"),
    )
    prepared.resolved_stream = resolved_stream
    return prepared


def physical_inputs(manifest, column, response_resolution):
    """Canonical physical identity shared by forward and grid-validation runs.

    Numerical opacity-table variants retain the absorber mapping, stellar/CIA
    inputs and physical conventions; their distinct archive pins are recorded
    separately in each run's asset provenance.
    """
    config = manifest.config
    star_spec = config["stellar"]
    star_hashes = {key: manifest.sha256[star_spec[key]]
                   for key in ("asset", "wavelength_asset", "flux_asset") if key in star_spec}
    return {
        "stellar_sha256": star_hashes, "stellar": star_spec,
        "gravity_cgs": column.settings.gravity_cgs,
        "internal_temperature": column.settings.internal_temperature,
        "mu0": column.settings.mu0,
        "mapping": config["opacity"]["mapping"],
        "cia_sha256": {name: manifest.sha256[spec["asset"]]
                       for name, spec in config["cia"].items()},
        "omitted_absorbers": config["opacity"]["omitted_absorbers"],
        "scattering": config["opacity"]["scattering"],
        "stellar_scattering_outside_range": column.settings.stellar_scattering_outside_range,
        "broadening_limitations": config["opacity"]["broadening_limitations"],
        "response_resolution": response_resolution,
        "thermochemistry": dict(column.entropy.thermodynamics.metadata),
    }


def _timed(function, *args):
    started = perf_counter()
    result = jax.block_until_ready(function(*args))
    return result, perf_counter() - started


def _serializable(value):
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return _serializable(np.asarray(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def provenance():
    root = Path(__file__).resolve().parents[1]
    import exogibbs

    def revision(path):
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    # Include uncommitted helper edits as well as the library revision.
    paths = sorted((root / "src" / "exojax").rglob("*.py"))
    paths += sorted((root / "examples").glob("*rce*.py"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.name != "ExoJAX_version.py"}
    gibbs_root = Path(exogibbs.__file__).resolve().parent
    gibbs_hashes = {str(path.relative_to(gibbs_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(gibbs_root.rglob("*.py"))}
    return {
        "exojax_revision": revision(root),
        "exogibbs_revision": revision(Path(exogibbs.__file__).resolve().parents[2]),
        "exogibbs_code_sha256": hashlib.sha256(json.dumps(gibbs_hashes, sort_keys=True).encode()).hexdigest(),
        "code_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "jax": jax.__version__, "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(), "devices": [d.device_kind for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }


def run_column(column, parameters, *, fixed_temperature=False, response_resolution=None,
               resolved_output=None, mask_continuation=False):
    """Run a single column; invalid solves never produce a predicted spectrum."""
    parameters = jnp.asarray(parameters, dtype=float)
    report = {"mode": "fixed_temperature" if fixed_temperature else "rce",
              "parameters": parameters}
    if column.settings.stellar_scattering:
        from _rce_wasp18b_audit import stellar_scattering_description

        report["stellar_scattering"] = stellar_scattering_description(column.settings)
    if fixed_temperature:
        temperature = jnp.asarray(column.solver_arguments[2])
        bottom = jnp.asarray(column.solver_arguments[3])
        if not bool(column.temperature_valid(temperature, bottom, parameters)):
            return dict(report, success=False, failure="outside_temperature_or_parameter_domain")
    else:
        if mask_continuation:
            from _rce_wasp18b_continuation import PreparedMaskContinuation

            print("Compiling the RCE solver with explicit stage initial states", flush=True)
            continuation = PreparedMaskContinuation(column, parameters)
            compile_time = continuation.compile_seconds
        else:
            started = perf_counter()
            print("Compiling the RCE solver", flush=True)
            compiled = jax.jit(column.solve).lower(parameters).compile()
            compile_time = perf_counter() - started
        print(f"RCE compilation finished in {compile_time:.2f} s; solving", flush=True)
        if mask_continuation:
            started = perf_counter()
            continued = continuation.run(parameters, on_stage=lambda stage: print(
                f"RCE stage {stage['index']}: {stage['status_name']}", flush=True))
            state, first_time = continued.state, perf_counter() - started
            report["mask_continuation"] = {
                "stages": continued.stages, "stop_reason": continued.stop_reason,
                "initial_state_policy": "Same explicit seed for each parameter point",
                "solver_options": continuation.solver_options,
                "max_stages": continuation.max_stages,
            }
        else:
            state, first_time = _timed(compiled, parameters)
        print(f"RCE status={int(state.status)}, converged={bool(state.converged)}, "
              f"elapsed={first_time:.2f} s", flush=True)
        warm_time = None
        if bool(state.converged) and not mask_continuation:
            _, warm_time = _timed(compiled, parameters)
        report.update(state=state._asdict(), solve_compile_seconds=compile_time,
                      first_solve_seconds=first_time,
                      warm_solve_seconds=warm_time)
        if not bool(state.converged):
            return dict(report, success=False, failure="rce_not_converged")
        temperature, bottom = state.temperature, state.bottom_temperature
    chemistry = jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2])
    report["chemistry"] = chemistry.diagnostics._asdict()
    if not bool(jnp.all(chemistry.diagnostics.valid)):
        return dict(report, success=False, failure="invalid_chemistry")
    column.check_response_coverage(response_resolution)
    started = perf_counter()
    stream = getattr(column, "resolved_stream", None)
    if resolved_output is not None and stream is None:
        raise ValueError("Saving the resolved spectrum requires --stream-resolved.")
    if stream is None:
        tables = column.resolved_opacity_tables()
        print("Compiling the resolved spectrum with explicit opacity arrays", flush=True)
        spectrum_fn = jax.jit(lambda t, tb, p, c, tables: column.spectrum(
            t, tb, p, c, resolving_power=response_resolution, resolved_tables=tables,
        )).lower(temperature, bottom, parameters, chemistry, tables).compile()
        spectrum_compile_time = perf_counter() - started
        prediction, first_spectrum_time = _timed(spectrum_fn, temperature, bottom, parameters, chemistry, tables)
        _, warm_spectrum_time = _timed(spectrum_fn, temperature, bottom, parameters, chemistry, tables)
        resolved_centers = (prediction if response_resolution is None else
                            column.spectrum(temperature, bottom, parameters, chemistry))
    else:
        print(f"Streaming {len(stream.chunks)} resolved spectral chunks", flush=True)
        top_flux = stream.top_flux(column, temperature, bottom, parameters, chemistry)
        prediction = jax.block_until_ready(stream.prediction(column, top_flux, response_resolution))
        first_spectrum_time = perf_counter() - started
        spectrum_compile_time, warm_spectrum_time = None, None
        resolved_centers = (prediction if response_resolution is None else
                            stream.prediction(column, top_flux))
        report["resolved_streaming"] = dict(stream.provenance,
            timing_scope="First spectrum includes chunk verification, disk reads, JIT compilation and transfer")
        if resolved_output is not None:
            path = Path(resolved_output).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as output:
                np.savez_compressed(output, nu_cm1=stream.nu,
                                    planet_upward_flux_per_cm1=top_flux,
                                    temperature_K=np.asarray(temperature),
                                    bottom_temperature_K=np.asarray(bottom),
                                    parameters=np.asarray(parameters))
            temporary.replace(path)
            report["resolved_spectrum_file"] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "units": "cm^-1; erg s^-1 cm^-2 (cm^-1)^-1; K",
            }
    spectral = jax.jit(column.fluxes)(temperature, bottom, parameters, chemistry)
    broadband = integrate_broadband_fluxes(spectral, column.g_weights, column.band_widths)
    if not bool(jnp.all(jnp.isfinite(prediction))) or not bool(jnp.all(jnp.isfinite(broadband.net))):
        return dict(report, success=False, failure="nonfinite_radiation")
    report.update(
        success=True, pressure_bar=column.pressure, temperature_K=temperature,
        bottom_temperature_K=bottom, broadband_flux=broadband._asdict(),
        bottom_incident_fraction=bottom_leakage_fraction(broadband),
        bottom_incident_over_internal_flux=broadband.incident[-1] / column.internal_flux,
        prediction=prediction, response_resolution=response_resolution,
        spectrum_compile_seconds=spectrum_compile_time,
        first_spectrum_seconds=first_spectrum_time, warm_spectrum_seconds=warm_spectrum_time,
    )
    if column.settings.stellar_scattering:
        from _rce_wasp18b_audit import stellar_flux_audit

        # Reuse the fixed-column audit to separate diffuse stellar light from
        # thermal downward flux at the black bottom. The solver is unchanged.
        report["stellar_scattering"]["stellar_flux_audit"] = stellar_flux_audit(
            column, temperature, parameters, chemistry)
    # Diagnostic only: band interpolation is not a resolved observation model.
    opa = next(iter(column.ckd.values()))
    if column.pixel_nu.min() >= opa.nu_bands[0] and column.pixel_nu.max() <= opa.nu_bands[-1]:
        planet = sample_ckd_bands_at_wavelengths(
            opa.nu_bands, column.g_weights @ spectral.upward[0], 1e4 / column.pixel_nu, unit="um"
        )
        star = jnp.asarray(column.stellar.at(np.asarray(column.pixel_nu)))
        ckd_prediction = pixel_flux_ratio(column.W, planet, star, column.area_ratio)
        report["ckd_interpolation_vs_resolved"] = spectral_error(
            ckd_prediction, resolved_centers, column.observation.uncertainty
        )
    if response_resolution is not None:
        report["response_vs_centers"] = spectral_error(
            prediction, resolved_centers, column.observation.uncertainty
        )
    check_temperatures = jnp.array([1500.0, 2500.0, 3500.0, 4500.0])
    planck_integrals = band_planck_flux(check_temperatures, column.band_nodes,
                                       column.band_weights) @ column.band_widths
    report["blackbody_integral"] = {
        "temperature_K": check_temperatures,
        "relative_error": planck_integrals / (SIGMA_SB * check_temperatures**4) - 1,
    }
    report["device_memory"] = jax.devices()[0].memory_stats()
    return report


def compare_runs(current, reference, *, kind="numerical"):
    """Compare converged runs with identical observations and parameter values."""
    if not current.get("success") or not reference.get("success"):
        raise ValueError("Both comparison runs must have succeeded.")
    for name in ("observation_sha256", "row_indices", "parameters", "mode"):
        if current[name] != reference[name]:
            raise ValueError(f"Comparison requires matching {name}.")
    # Before this explicit policy existed, out-of-range scattering was always
    # rejected. Interpret missing legacy fields as that unchanged default.
    identities = [{"stellar_scattering_outside_range": "error", **run["physical_inputs"]}
                  for run in (current, reference)]
    if kind == "numerical" and identities[0] != identities[1]:
        raise ValueError("Numerical convergence requires identical physical inputs; use a physics comparison.")
    result = spectral_error(current["prediction"], reference["prediction"], current["uncertainty"])
    result["kind"] = kind
    p, pr = np.asarray(current["pressure_bar"]), np.asarray(reference["pressure_bar"])
    common = (p >= pr[0]) & (p <= pr[-1])
    if not np.any(common):
        raise ValueError("Pressure grids have no common interval for comparison.")
    reference_temperature = np.interp(np.log(p[common]), np.log(pr), reference["temperature_K"])
    result["max_temperature_change_K"] = float(np.max(np.abs(
        np.asarray(current["temperature_K"])[common] - reference_temperature
    )))
    result["top_net_flux_change_cgs"] = (current["broadband_flux"]["net"][0]
                                          - reference["broadband_flux"]["net"][0])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--fixed-temperature", action="store_true")
    parser.add_argument("--mask-continuation", action="store_true",
                        help="Use bounded active-mask restarts within each parameter point.")
    parser.add_argument("--stream-resolved", action="store_true",
                        help="Load pinned resolved opacity chunks after the RCE solve.")
    parser.add_argument("--resolved-spectrum-output", type=Path,
                        help="Save the complete streamed emergent spectrum for response comparisons.")
    parser.add_argument("--response-resolution", type=float)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--comparison-kind", choices=("numerical", "physics"), default="numerical")
    args = parser.parse_args(argv)
    if args.resolved_spectrum_output is not None and not args.stream_resolved:
        parser.error("--resolved-spectrum-output requires --stream-resolved")
    if args.mask_continuation and args.fixed_temperature:
        parser.error("--mask-continuation requires an RCE solve")
    jax.config.update("jax_enable_x64", True)
    report = {"schema_version": "exojax.rce-wasp18b-run/1", "success": False,
              "p7_physical_acceptance": "not_certified"}
    try:
        observation = load_emission_candidate(args.observation)
        report.update(observation_sha256=observation.sha256, row_indices=observation.row_indices,
                      uncertainty=observation.uncertainty, observation_status="candidate")
        print("Verifying the pinned input assets", flush=True)
        manifest = load_manifest(args.manifest)
        report.update(asset_sha256=manifest.sha256, configuration=manifest.config,
                      provenance=provenance())
        config = manifest.config
        parameters = np.asarray(config["parameters"], dtype=float)
        lower, upper = np.asarray(config["prior_lower"]), np.asarray(config["prior_upper"])
        if (parameters.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(parameters) & np.isfinite(lower) & np.isfinite(upper))
                or not np.all((lower < upper) & (parameters >= lower) & (parameters <= upper))):
            raise ValueError("Supply three finite parameters inside explicit ordered prior bounds.")
        print("Loading opacity tables and preparing the column", flush=True)
        column = prepare_column(manifest, observation, stream_resolved=args.stream_resolved)
        print("Column preparation finished", flush=True)
        if not bool(column.temperature_valid(jnp.asarray(column.solver_arguments[2]),
                                             column.solver_arguments[3], jnp.asarray(parameters))):
            raise ValueError("Initial temperature or parameters exceed the common physical domain.")
        report.update(temperature_domain_K=column.temperature_range,
                      settings=asdict(column.settings), species=column.chemistry.species,
                      thermodynamics=column.entropy.thermodynamics.metadata,
                      physical_inputs=physical_inputs(manifest, column, args.response_resolution))
        if args.check_inputs:
            report.update(success=True, mode="check_inputs")
        else:
            report.update(run_column(column, parameters, fixed_temperature=args.fixed_temperature,
                                     response_resolution=args.response_resolution,
                                     resolved_output=args.resolved_spectrum_output,
                                     mask_continuation=args.mask_continuation))
        report = _serializable(report)
        if args.compare and report.get("success"):
            report["comparison"] = compare_runs(report, json.loads(args.compare.read_text()),
                                                 kind=args.comparison_kind)
        elif args.compare:
            report["comparison_skipped"] = "The forward solve failed; no spectrum comparison is valid."
    except (ValueError, KeyError, FileNotFoundError) as error:
        report.update(success=False, failure=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_serializable(report), indent=2, allow_nan=False) + "\n")
    print(f"{'OK' if report['success'] else 'FAILED'}: {args.output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
