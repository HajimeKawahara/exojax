"""Fixed-column neutral-atom broadening using paired native line spectra.

Both cases use the same actual temperatures and native teacher. Effective
pressure changes only the native collision width; gas columns, chemistry,
other opacities and sources retain physical pressure. No RCE is solved.
"""

import argparse
from copy import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import prepare_rce_wasp18b_atomic as atomic
from _rce_broadband import Fluxes, irradiation_flux
from _rce_observation import load_emission_candidate
from _rce_wasp18b import continuum_optical_depth
from _rce_wasp18b_assets import _sha256, load_manifest
from rce_wasp18b_forward import _serializable, prepare_column, provenance
from exojax.opacity.ckd.core import compute_ckd_from_xsmatrix, gauss_legendre_grid
from exojax.opacity.ckd.mixing import mix_ckd_rorr
from exojax.utils.constants import bar_cgs, m_u


def effective_pressure(pressure, fractions, fixed):
    """Use all-particle H/He/H2 fractions, without renormalizing or clamping."""
    pressure, fractions, fixed = map(np.asarray, (pressure, fractions, fixed))
    if (pressure.ndim != 1 or fractions.shape != (pressure.size, 3) or fixed.shape != (3,)
            or not np.all(np.isfinite(pressure) & (pressure > 0.))
            or not np.all(np.isfinite(fractions) & (fractions >= 0.))
            or not np.all(np.isfinite(fixed) & (fixed >= 0.))
            or np.any(fractions.sum(axis=1) > 1. + 1e-12)):
        raise ValueError("Use positive pressures and nonnegative all-particle H/He/H2 fractions")
    denominator = float(fixed[0] + .41336 * fixed[1] + .85 * fixed[2])
    if denominator <= 0.:
        raise ValueError("The fixed broadening factor must be positive")
    factor = fractions[:, 0] + .41336 * fractions[:, 1] + .85 * fractions[:, 2]
    return pressure * (factor / denominator)


def paired_native_ckd(teacher, temperature, pressure, effective, edges, ng):
    """Reuse the frozen four-layer wrapper and the original inclusive band masks."""
    layers = len(temperature)
    values = teacher.xsmatrix(jnp.tile(jnp.asarray(temperature), 2),
                             jnp.asarray(np.concatenate((pressure, effective))))
    ggrid, _ = gauss_legendre_grid(ng)
    bands, counts = [], []
    for lower, upper in edges:
        mask = (teacher.nu_grid >= lower) & (teacher.nu_grid <= upper)
        if not np.any(mask):
            raise ValueError("A native band has no spectral samples")
        counts.append(int(mask.sum()))
        bands.append(np.asarray(compute_ckd_from_xsmatrix(values[:, mask], ggrid)))
    logk = np.stack(bands, axis=-1).reshape(2, layers, ng, len(edges))
    if not np.all(np.isfinite(logk)):
        raise ValueError("Nonfinite native atomic CKD coefficients")
    return logk, counts


def native_chunk(plan, index, temperature, pressure, effective):
    """Read the exact frozen line selection and release its compiled executable."""
    from exojax.database.kurucz.api import AdbKurucz

    spec, physics = plan["settings"], plan["atomic_preparation"]
    species = plan["species"]
    grid, core, selection, edges = atomic.base.chunk_grid(plan, index)
    padding = physics["line_padding_cm1"]
    adb = AdbKurucz(Path(plan["mdb_path"]),
        nurange=[max(0., grid[0] - padding), grid[-1] + padding],
        gpu_transfer=False, vmr_fraction=physics["perturber_fractions_H_He_H2"])
    if (np.any(adb._ielem != atomic.ATOMIC_LINES[species][0]) or np.any(adb._iion != 1)
            or np.any(adb._elower < 0.) or np.any(adb._eupper < 0.)
            or np.min(temperature) < np.min(adb.T_gQT)
            or np.max(temperature) > np.max(adb.T_gQT)):
        raise ValueError("The native atomic line identity, energies or partition domain differ")
    record = json.loads((Path(plan["output_dir"]) / species / f"chunk_{index:05d}" / "complete.json").read_text())
    if (record["grid_fingerprint"] != plan["grid_fingerprint"]
            or record["provenance"]["number_of_lines"] != len(adb.nu_lines)
            or record["provenance"]["atomic_preparation"] != physics):
        raise ValueError("The original chunk and current native teacher differ")
    atomic.base._check_sources(record["provenance"]["sources"])
    teacher, cropped = None, None
    try:
        if len(adb.nu_lines):
            adb.generate_jnp_arrays()
            teacher = atomic.BatchedAtomicTeacher(adb, grid, species, physics["line_profile"], physics["line_batch_size"])
        cropped = atomic.base.CroppedTeacher(teacher, core, selection, record["provenance"])
        values, counts = paired_native_ckd(cropped, temperature, pressure, effective, edges, spec["ng"])
        metadata = dict(number_of_lines=len(adb.nu_lines), teacher_points=len(grid), core_points=len(core),
                        band_sample_counts=counts, original_chunk_provenance=record["provenance"])
        return values, edges, metadata
    finally:
        if teacher is not None:
            teacher._compiled_xsmatrix.clear_cache()
        # The jitted bound method forms a cycle with its teacher. Do not retain
        # hundreds of line arrays/executables across the full broadband loop.
        del cropped, teacher, adb
        gc.collect()


def cached_native_chunk(directory, fingerprint, index, calculate):
    """Resume only verified paired coefficients with the exact input identity."""
    directory.mkdir(parents=True, exist_ok=True)
    coefficients = directory / f"chunk_{index:05d}.npz"
    completion = directory / f"chunk_{index:05d}.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["fingerprint"] != fingerprint or record["sha256"] != _sha256(coefficients):
            raise ValueError("Changed broadening-audit checkpoint; use a new output directory")
        with np.load(coefficients, allow_pickle=False) as saved:
            return saved["logk"], saved["edges"], record
    started = perf_counter()
    logk, edges, metadata = calculate()
    np.savez(coefficients, logk=logk, edges=edges)
    record = dict(fingerprint=fingerprint, sha256=_sha256(coefficients), chunk=index,
                  native_seconds=perf_counter() - started, **metadata)
    atomic.base._write_json(completion, record)
    return logk, edges, record


def mix_with_atom(depths, cross_section, amount, continuum, weights, index):
    """Replace one atom while preserving the original species order in RORR."""
    changed = depths.at[index].set(cross_section * amount[:, None, None])
    return mix_ckd_rorr(changed, weights) + continuum[:, None, :]


def flux_at_depth(column, depth, nodes, star, temperature, bottom, parameters, chemistry):
    """Reuse the production fluxes, including the selected stellar scattering."""
    local = copy(column)
    local.band_nodes, local.star_bands = nodes, star
    local.band_widths = jnp.ones(nodes.shape[0])  # Shape only inside fluxes.
    local.optical_depth = lambda *args, **kwargs: depth
    return local.fluxes(temperature, bottom, parameters, chemistry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "observation", "state", "atomic-preparation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", help="Named state in an initialization diagnostic")
    parser.add_argument("--chunk-index", type=int, action="append", help="Limit to smoke chunks; omitted means all chunks")
    args = parser.parse_args(argv)
    if args.output.resolve() in {path.resolve() for path in (args.manifest, args.observation, args.state, args.atomic_preparation)}:
        parser.error("The output must be separate from every input")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    report = dict(status="loading", certifies_p7=False, started_utc=datetime.now(timezone.utc).isoformat(),
        temperature_and_composition_fixed=True, pressure_clamping=False, stark_electron_density_cm3=0.,
        limitations=["Selected neutral-atom collision-width sensitivity, not an error bound for physical broadening models.",
                     "Other atoms, molecular broadening, nonzero Stark width, RCE re-equilibration and resolved P1 prediction are not assessed in this run.",
                     "Both paired cases use the actual temperatures. Stored-table differences separately include interpolation."])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic.base._write_json(args.output, _serializable(report))

    try:
        source = json.loads(args.state.read_text())
        if args.case:
            state = next(item["state"] for item in source["results"] if item["name"] == args.case)
        else:
            state = source["state"]
        if not all(state[key] for key in ("converged", "domain_valid", "physics_valid")):
            raise ValueError("Use a converged valid input state")
        complete = json.loads(args.atomic_preparation.read_text())
        plan, physics = complete["plan"], complete["plan"]["atomic_preparation"]
        atom = plan["species"]
        if (atom not in atomic.ATOMIC_LINES
                or physics["driver_sha256"] != _sha256(Path(atomic.__file__))
                or physics["base_script_sha256"] != _sha256(Path(atomic.base.__file__))):
            raise ValueError("Use the unchanged native atomic preparation driver and plan")
        manifest, observation = load_manifest(args.manifest), load_emission_candidate(args.observation)
        atom_asset = manifest.config["opacity"]["ckd"][atom]["asset"]
        if manifest.sha256[atom_asset] != complete["files"]["ckd.npz"]["sha256"]:
            raise ValueError("Manifest atom differs from the original preparation")
        if (not np.array_equal(source["parameters"], manifest.config["parameters"])
                or source["observation_sha256"] != observation.sha256):
            raise ValueError("State and manifest parameters or observation differ")
        if not args.case and (source["asset_sha256"] != manifest.sha256
                              or source["configuration"] != manifest.config):
            raise ValueError("State and manifest physical inputs differ")
        column = prepare_column(manifest, observation, stream_resolved=True)
        temperature, bottom = jnp.asarray(state["temperature"]), jnp.asarray(state["bottom_temperature"])
        parameters, pressure = jnp.asarray(source["parameters"]), np.asarray(column.pressure)
        if (not np.array_equal(source["pressure_bar"], pressure)
                or not bool(column.temperature_valid(temperature, bottom, parameters))):
            raise ValueError("State and pressure/temperature domain differ")
        chemistry = jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2])
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            raise ValueError("Invalid recomputed chemistry")
        species = tuple(column.ckd)
        fractions = np.asarray(chemistry.x[:-1])[:, [column.chemistry.species.index(name) for name in ("H1", "He1", "H2")]]
        effective = effective_pressure(pressure, fractions, physics["perturber_fractions_H_He_H2"])
        original_sources = atomic._source_files(Path(plan["mdb_path"]))
        code_paths = [Path(__file__), Path(atomic.__file__), Path(atomic.base.__file__),
                      Path(__file__).with_name("_rce_wasp18b.py"), Path(__file__).with_name("_rce_broadband.py"),
                      Path(__file__).with_name("_rce_wasp18b_scattering.py")]
        import exojax.opacity.ckd.core as ckd_core
        import exojax.opacity.ckd.mixing as ckd_mixing
        import exojax.rt.flux as flux_module
        code_paths += [Path(module.__file__) for module in (ckd_core, ckd_mixing, flux_module)]
        identity = dict(state_sha256=_sha256(args.state), atomic_preparation_sha256=_sha256(args.atomic_preparation),
                        asset_sha256=manifest.sha256, code_sha256={str(p.resolve()): _sha256(p) for p in code_paths},
                        native_sources=original_sources, case=args.case)
        fingerprint = atomic.base._digest_json(identity)
        report.update(identity=identity, fingerprint=fingerprint, provenance=provenance(), species=species,
            audited_atom=atom, atomic_line_profile=physics["line_profile"],
            parameters=parameters, temperature_K=temperature, bottom_temperature_K=bottom, pressure_bar=pressure,
            effective_pressure_bar=effective, fractions_H_He_H2=fractions, chemistry=chemistry.diagnostics._asdict(),
            fixed_perturbers_H_He_H2=physics["perturber_fractions_H_He_H2"], configuration=manifest.config)
        amount = (jnp.diff(column.boundaries) * bar_cgs / (column.settings.gravity_cgs * m_u * chemistry.mmw[:-1]))
        amounts = {name: amount * chemistry.x[:-1, column.chemistry.species.index(name)] * column.isotope_fractions[name] for name in species}
        stored_xs = {name: jax.jit(opa.xstensor_ckd)(temperature, column.pressure) for name, opa in column.ckd.items()}
        depths = jnp.stack([stored_xs[name] * amounts[name][:, None, None] for name in species])
        stored_atom = stored_xs[atom]
        del stored_xs
        continuum = jax.jit(lambda t, c: continuum_optical_depth(column.band_nodes.ravel(), t, column.pressure,
            column.boundaries, c, column.chemistry.species, column.settings.gravity_cgs, column.cia))(temperature, chemistry)
        continuum = jnp.sum(continuum.reshape((len(pressure),) + column.band_nodes.shape) * column.band_weights, axis=-1)
        horizontal = irradiation_flux(column.star_bands, column.a_over_rstar, jnp.exp(parameters[2]))
        incident = float(jnp.sum(horizontal * column.band_widths))

        @jax.jit
        def calculate(d, cont, xs, nodes, star):
            depth = mix_with_atom(d, xs, amounts[atom], cont, column.g_weights, species.index(atom))
            return flux_at_depth(column, depth, nodes, star, temperature, bottom, parameters, chemistry)

        totals = {name: np.zeros((4, len(pressure) + 1)) for name in ("native_fixed", "native_actual", "stored_fixed")}
        indices = sorted(set(args.chunk_index)) if args.chunk_index is not None else list(range(plan["number_of_chunks"]))
        if not indices or any(i < 0 or i >= plan["number_of_chunks"] for i in indices):
            raise ValueError("Invalid audit chunk selection")
        report.update(status="native_chunks", selected_chunks=indices, chunks=[], full_broadband=len(indices) == plan["number_of_chunks"])
        save()
        for index in indices:
            print(f"Auditing native {atom} chunk {index + 1}/{plan['number_of_chunks']}", flush=True)
            logk, edges, record = cached_native_chunk(args.output.with_suffix(".chunks"), fingerprint, index,
                lambda: native_chunk(plan, index, np.asarray(temperature), pressure, effective))
            first = index * plan["settings"]["bands_per_chunk"]
            selection = slice(first, first + len(edges))
            if not np.array_equal(edges, np.asarray(column.band_edges[selection])):
                raise ValueError("Native and column band edges differ")
            for name, xs in zip(totals, (np.exp(logk[0]), np.exp(logk[1]), stored_atom[..., selection])):
                flux = np.asarray(calculate(depths[..., selection], continuum[:, selection], xs,
                                            column.band_nodes[selection], column.star_bands[selection]))
                if not np.all(np.isfinite(flux)):
                    raise ValueError("Nonfinite paired native atomic boundary flux")
                totals[name] += np.einsum('flgb,g,b->fl', flux, column.g_weights, column.band_widths[selection])
            report["chunks"].append(dict(index=index, native_seconds=record["native_seconds"],
                number_of_lines=record["number_of_lines"], teacher_points=record["teacher_points"],
                checkpoint_sha256=record["sha256"], band_sample_counts=record["band_sample_counts"]))
            report["boundary_flux_cgs"] = {name: dict(zip(Fluxes._fields, value)) for name, value in totals.items()}
            save()
        comparisons = {}
        for name, left, right in (("native_actual_minus_native_fixed", "native_actual", "native_fixed"),
                                  ("native_fixed_minus_stored_fixed", "native_fixed", "stored_fixed")):
            delta = totals[left] - totals[right]
            comparisons[name] = dict(boundary_difference_cgs=dict(zip(Fluxes._fields, delta)),
                max_boundary_difference_over_incident=dict(zip(Fluxes._fields, np.max(np.abs(delta), axis=1) / incident)),
                layer_heating_difference_cgs=np.diff(delta[-1]),
                max_layer_heating_difference_over_incident=float(np.max(np.abs(np.diff(delta[-1])))) / incident)
        report.update(status="complete_diagnostic_only", comparisons=comparisons, incident_flux_for_normalization_cgs=incident,
                      device_memory_stats=jax.devices()[0].memory_stats())
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report.update(seconds=perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
