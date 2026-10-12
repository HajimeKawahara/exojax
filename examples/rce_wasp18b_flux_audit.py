"""Compare CKD and resolved boundary-flux integrals on a common interval.

Use only complete CKD bands inside the resolved table coverage. This diagnostic
does not compare band-center interpolation in observation space, and does not
certify the full broadband RCE heating calculation. Resolved coefficients are
verified and loaded one prepared spectral chunk at a time.
"""

import argparse
from copy import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import Fluxes, integrate_broadband_fluxes
from _rce_observation import load_emission_candidate
from _rce_wasp18b_assets import _sha256, load_manifest
from _rce_wasp18b_streaming import _verify
from rce_wasp18b_forward import _serializable, prepare_column, provenance
from exojax.opacity import OpaDiffgrid


def common_band_interval(band_edges, resolved_nu):
    """Select contiguous complete bands, without integrating partial CKD bins."""
    edges, nu = np.asarray(band_edges), np.asarray(resolved_nu)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or nu.ndim != 1 or nu.size < 2
            or not np.all(np.isfinite(edges)) or not np.all(np.isfinite(nu))
            or np.any(np.diff(nu) <= 0) or np.any(edges[:, 1] <= edges[:, 0])
            or not np.allclose(edges[1:, 0], edges[:-1, 1], rtol=1e-12, atol=0.)):
        raise ValueError("Use finite contiguous bands and increasing resolved coordinates")
    indices = np.flatnonzero((edges[:, 0] >= nu[0]) & (edges[:, 1] <= nu[-1]))
    if not indices.size:
        raise ValueError("No complete CKD band lies inside the resolved coverage")
    return indices, (float(edges[indices[0], 0]), float(edges[indices[-1], 1]))


def integrate_interval(nu, values, lower, upper):
    """Integrate the last axis, interpolating exactly two requested endpoints."""
    nu, values = np.asarray(nu), np.asarray(values)
    if (nu.ndim != 1 or nu.size < 2 or values.shape[-1] != nu.size
            or np.any(np.diff(nu) <= 0) or not np.all(np.isfinite(nu))
            or not np.all(np.isfinite(values)) or not nu[0] <= lower < upper <= nu[-1]):
        raise ValueError("Integration endpoints must lie inside a finite increasing grid")

    def endpoint(point):
        index = int(np.clip(np.searchsorted(nu, point, side="right") - 1, 0, nu.size - 2))
        weight = (point - nu[index]) / (nu[index + 1] - nu[index])
        return values[..., index] * (1. - weight) + values[..., index + 1] * weight

    inside = (nu > lower) & (nu < upper)
    grid = np.concatenate(([lower], nu[inside], [upper]))
    samples = np.concatenate((endpoint(lower)[..., None], values[..., inside], endpoint(upper)[..., None]), axis=-1)
    return np.sum(np.diff(grid) * (samples[..., :-1] + samples[..., 1:]) * .5, axis=-1)


def select_ckd_bands(column, indices):
    """Make a small column view while retaining the original chemistry and RT."""
    local = copy(column)
    local.ckd = {}
    for name, original in column.ckd.items():
        opa = copy(original)
        info = original.ckd_info
        opa.ckd_info = replace(info, log_kggrid=info.log_kggrid[..., indices],
                               nu_bands=info.nu_bands[indices], band_edges=info.band_edges[indices])
        opa.nu_bands, opa.band_edges = opa.ckd_info.nu_bands, opa.ckd_info.band_edges
        local.ckd[name] = opa
    for name in ("band_edges", "band_nodes", "band_widths", "star_bands"):
        setattr(local, name, getattr(column, name)[indices])
    return local


def resolved_boundary_integrals(column, temperature, bottom, parameters, chemistry, interval):
    """Stream every Fluxes component and sum trapezoids across shared seams."""
    plan = column.resolved_stream
    if (tuple(column.lbl) != plan.species or not np.array_equal(np.asarray(column.nu), plan.nu)
            or not bool(column.temperature_valid(temperature, bottom, parameters))
            or not bool(jnp.all(chemistry.diagnostics.valid))):
        raise ValueError("Streaming integration requires matching tables and a valid physical state")
    lower, upper = interval
    if not plan.nu[0] <= lower < upper <= plan.nu[-1]:
        raise ValueError("The common interval exceeds resolved coverage")
    plan.provenance["all_chunk_coefficients_verified"] = False

    @jax.jit
    def calculate(nu, stellar, tables, temperature, bottom, parameters, chemistry):
        local = copy(column)
        local.nu, local.star_resolved = nu, stellar
        return local.fluxes(temperature, bottom, parameters, chemistry,
                            resolved=True, resolved_tables=tables)

    total = {name: np.zeros(len(column.boundaries)) for name in Fluxes._fields}
    previous_nu, previous_endpoint = None, None
    maximum_seam_difference = {name: 0. for name in Fluxes._fields}
    used_chunks = 0
    for index, (chunk, nu) in enumerate(zip(plan.chunks, plan.grids)):
        tables = {}
        for name in plan.species:
            entry = chunk[name]
            _verify(entry.path, entry.sha256)
            _verify(entry.metadata_path, entry.metadata_sha256)
            opa = OpaDiffgrid.from_saved_opa(str(entry.path))
            if not np.array_equal(opa.nu_grid, nu):
                raise ValueError("Resolved chunk grid changed after setup")
            opa.check_pressure_grid(np.asarray(column.pressure))
            info = opa.diffgrid_info
            tables[name] = (info.inverse_temperature_grid, info.log_cross_section_grid,
                            info.log_cross_section_derivative_grid, info.log_cross_section_floor)
        flux = calculate(jnp.asarray(nu), jnp.asarray(column.stellar.at(nu)), tables,
                         temperature, bottom, parameters, chemistry)
        values = {name: np.asarray(value).copy() for name, value in flux._asdict().items()}
        if not all(np.all(np.isfinite(value)) for value in values.values()):
            raise ValueError("Nonfinite resolved boundary flux")
        if previous_endpoint is not None:
            if nu[0] != previous_nu:
                raise ValueError("Resolved chunks must share exactly one endpoint")
            for name, value in values.items():
                maximum_seam_difference[name] = max(maximum_seam_difference[name],
                    float(np.max(np.abs(value[:, 0] - previous_endpoint[name]))))
                # Match the merged archive and the existing top-flux streamer.
                value[:, 0] = previous_endpoint[name]
        previous_nu = nu[-1]
        previous_endpoint = {name: value[:, -1].copy() for name, value in values.items()}
        start, stop = max(lower, nu[0]), min(upper, nu[-1])
        if start < stop:
            for name, value in values.items():
                total[name] += integrate_interval(nu, value, start, stop)
            used_chunks += 1
        del tables, opa, info, flux, values
        if (index + 1) % 8 == 0 or index + 1 == len(plan.chunks):
            print(f"Integrated resolved chunk {index + 1}/{len(plan.chunks)}", flush=True)
    plan.provenance["all_chunk_coefficients_verified"] = True
    return total, dict(chunk_count_used=used_chunks,
        maximum_raw_seam_flux_density_difference_cgs=maximum_seam_difference,
        seam_policy="Use the preceding chunk's endpoint, as in the merged resolved archive; integrate each adjacent interval once.")


def compare_boundary_integrals(ckd, resolved):
    """Normalize differences by incident energy in the same spectral interval."""
    incident = float(ckd["incident"][0])
    if not np.isfinite(incident) or incident <= 0:
        raise ValueError("The common interval needs positive incident stellar flux")
    output = {}
    for name in Fluxes._fields:
        left, right = np.asarray(ckd[name]), np.asarray(resolved[name])
        if left.ndim != 1 or left.shape != right.shape or not np.all(np.isfinite(left) & np.isfinite(right)):
            raise ValueError("All boundary integrals must be matching finite vectors")
        difference = left - right
        output[name] = dict(ckd_cgs=left, resolved_cgs=right, ckd_minus_resolved_cgs=difference,
            difference_over_interval_incident=difference / incident,
            max_absolute_difference_over_interval_incident=float(np.max(np.abs(difference)) / incident))
    divergence = np.diff(output["net"]["ckd_minus_resolved_cgs"])
    return dict(interval_incident_flux_cgs=incident, fluxes=output,
        layer_net_flux_divergence_difference_cgs=divergence,
        max_absolute_layer_divergence_difference_over_interval_incident=float(np.max(np.abs(divergence)) / incident),
        divergence_convention="Net upward flux at the lower boundary minus net upward flux at the upper boundary.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "observation", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.manifest.resolve(), args.observation.resolve(), args.state.resolve()}:
        parser.error("The output must be separate from every input")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="loading",
        physical_validation_complete=False, certifies_full_broadband_rce=False,
        scope="Fixed converged column; CKD versus resolved boundary-flux integrals on complete common bands. This is not band-center interpolation in observation space.",
        source_sha256={str(path.resolve()): _sha256(path.resolve()) for path in (
            Path(__file__), Path(__file__).with_name("_rce_wasp18b.py"),
            Path(__file__).with_name("_rce_broadband.py"), Path(__file__).with_name("_rce_wasp18b_streaming.py"))})

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(_serializable(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    try:
        source = json.loads(args.state.read_text())
        state = source["state"] if "state" in source else source["stages"][-1]["state"]
        if not all(state[key] for key in ("converged", "domain_valid", "physics_valid")):
            raise ValueError("Use a converged valid RCE state")
        manifest = load_manifest(args.manifest)
        observation = load_emission_candidate(args.observation)
        if not np.array_equal(source["parameters"], manifest.config["parameters"]):
            raise ValueError("State and manifest parameters differ")
        if source.get("observation_sha256", observation.sha256) != observation.sha256:
            raise ValueError("State and observation hashes differ")
        report.update(asset_sha256=manifest.sha256, provenance=provenance(),
            input_state=dict(path=str(args.state.resolve()), sha256=_sha256(args.state)),
            input_manifest=str(args.manifest.resolve()), observation_sha256=observation.sha256)
        print("Preparing the column with metadata-only resolved tables", flush=True)
        column = prepare_column(manifest, observation, stream_resolved=True)
        temperature, bottom = jnp.asarray(state["temperature"]), jnp.asarray(state["bottom_temperature"])
        parameters = jnp.asarray(source["parameters"])
        chemistry = jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2])
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            raise ValueError("Invalid recomputed chemistry")
        indices, interval = common_band_interval(column.band_edges, column.nu)
        report.update(parameters=parameters, temperature_K=temperature, bottom_temperature_K=bottom,
            pressure_boundaries_bar=column.boundaries, chemistry_diagnostics=chemistry.diagnostics._asdict(),
            common_interval_cm1=interval, ckd_band_indices=indices, ckd_band_count=len(indices),
            resolved_coverage_cm1=[float(column.nu[0]), float(column.nu[-1])],
            status="ckd", setup_seconds=perf_counter() - started)
        save()
        local = select_ckd_bands(column, indices)
        begin = perf_counter()
        ckd = integrate_broadband_fluxes(local.fluxes(temperature, bottom, parameters, chemistry),
                                        local.g_weights, local.band_widths)
        ckd = {name: np.asarray(value) for name, value in ckd._asdict().items()}
        report.update(ckd_seconds=perf_counter() - begin, status="resolved")
        save()
        del local
        jax.clear_caches()
        begin = perf_counter()
        resolved, integration = resolved_boundary_integrals(column, temperature, bottom, parameters, chemistry, interval)
        report.update(resolved_seconds=perf_counter() - begin, integration=integration,
            comparison=compare_boundary_integrals(ckd, resolved), resolved_streaming=column.resolved_stream.provenance,
            device_memory_stats=jax.devices()[0].memory_stats(), status="complete_diagnostic_only")
        full_incident = float(jnp.sum(column.star_bands * column.band_widths)
                              * jnp.exp(parameters[2]) / column.a_over_rstar**2)
        report["full_broadband_incident_flux_cgs"] = full_incident
        report["comparison"]["interval_fraction_of_full_incident"] = report["comparison"]["interval_incident_flux_cgs"] / full_incident
        for row in report["comparison"]["fluxes"].values():
            row["max_absolute_difference_over_full_broadband_incident"] = float(np.max(np.abs(row["ckd_minus_resolved_cgs"])) / full_incident)
        report["limitations"] = [
            "Only the common resolved interval is audited; ultraviolet and far-infrared heating remain outside this comparison.",
            "Both paths inherit the same line inventory, isotopologue assumptions and pressure-broadening limitations.",
            "The resolved reference shares the prepared temperature grid and finite line halo; it is not a line-completeness certificate.",
            "Incident-flux differences also measure stellar sampling quadrature; no observed-bin error is inferred from these integrals.",
        ]
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report.update(total_seconds=perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
