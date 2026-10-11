"""Measure selected molecular additions on a fixed P7 atmospheric column.

The frozen molecular loader evaluates only the actual layer T/P pairs. Small
native CKD and observed-range cross-section checkpoints replace full T/P grids.
These are conditional sensitivities, not global omission bounds or RCE solves.
"""

import argparse
from copy import copy, deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import prepare_rce_wasp18b_supplemental as supplemental
import prepare_rce_wasp18b_hcn_partition as hcn_partition
from _rce_broadband import Fluxes, irradiation_flux
from _rce_observation import load_emission_candidate
from _rce_wasp18b import continuum_optical_depth, spectral_error
from _rce_wasp18b_assets import _sha256, load_manifest
from _rce_wasp18b_streaming import _verify
from rce_wasp18b_forward import _serializable, prepare_column, provenance
from exojax.opacity import OpaDiffgrid
from exojax.opacity.ckd.core import compute_ckd_from_xsmatrix, gauss_legendre_grid
from exojax.opacity.ckd.mixing import mix_ckd_rorr
from exojax.utils.constants import bar_cgs, m_u


base = supplemental.base
COMPONENTS = {
    "H1O1": dict(chemical_species=("H1O1",), line_list="OH/16O-1H/MYTHOS"),
    "Fe1H1": dict(chemical_species=("Fe1H1",), line_list="FeH/56Fe-1H/MoLLIST"),
    "O1V1": dict(chemical_species=("O1V1",), line_list="VO/51V-16O/VOMYT"),
    "HCN_HNC": dict(chemical_species=("C1H1N1_hcn", "C1H1N1_hnc"),
                    line_list="HCN/1H-12C-14N/Harris"),
}
SPECIES = tuple(COMPONENTS)


def component_fractions(names, chemistry, chemical_species):
    """Count every chemical species once, including both members of HCN/HNC."""
    used, fractions = set(), {}
    for name in names:
        members = COMPONENTS[name]["chemical_species"]
        if used.intersection(members):
            raise ValueError("Overlapping molecular components would double-count populations")
        used.update(members)
        indices = [chemical_species.index(member) for member in members]
        fractions[name] = jnp.sum(jnp.take(chemistry.x[:-1], jnp.asarray(indices), axis=-1), axis=-1)
    return fractions


def component_database(name, database_root, hcn_hnc_asset=None):
    """An HCN/HNC calculation requires the explicitly derived, pinned PF."""
    descriptor = dict(COMPONENTS[name])
    if name == "HCN_HNC":
        if hcn_hnc_asset is None:
            raise ValueError("HCN_HNC requires --hcn-hnc-asset from the direct-state partition preparer")
        database = Path(hcn_hnc_asset).expanduser().resolve()
        descriptor["partition_asset"] = hcn_partition.verify_asset(database)
        descriptor["partition_metadata_sha256"] = _sha256(database / "direct_partition.json")
    else:
        database = (Path(database_root) / descriptor["line_list"]).expanduser().resolve()
    return database, descriptor


def fixed_teacher_values(teacher, temperature, pressure, edges, ng, resolved_range, floor):
    """Evaluate paired layer coordinates once and retain native band masks."""
    temperature, pressure = np.asarray(temperature), np.asarray(pressure)
    if (temperature.ndim != 1 or temperature.shape != pressure.shape
            or not np.all(np.isfinite(temperature) & (temperature > 0.))
            or not np.all(np.isfinite(pressure) & (pressure > 0.))):
        raise ValueError("Use aligned finite positive layer temperatures and pressures")
    edges = np.asarray(edges)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or not np.all(np.isfinite(edges)) or np.any(edges[:, 0] >= edges[:, 1])
            or not np.isfinite(floor) or floor <= 0.):
        raise ValueError("Use ordered finite band edges and a positive cross-section floor")
    values = np.asarray(teacher.xsmatrix(jnp.asarray(temperature), jnp.asarray(pressure)))
    nu = np.asarray(teacher.nu_grid)
    if values.shape != (len(temperature), len(nu)) or not np.all(np.isfinite(values)):
        raise ValueError("Native cross sections have an invalid shape or nonfinite values")
    ggrid, _ = gauss_legendre_grid(ng)
    bands, counts = [], []
    for lower, upper in edges:
        selected = (nu >= lower) & (nu <= upper)
        if not selected.any():
            raise ValueError("A native band has no spectral samples")
        bands.append(np.asarray(compute_ckd_from_xsmatrix(jnp.asarray(values[:, selected]), ggrid)))
        counts.append(int(selected.sum()))
    observed = (nu >= resolved_range[0]) & (nu <= resolved_range[1])
    return dict(logk=np.stack(bands, axis=-1), edges=np.asarray(edges),
                resolved_nu=nu[observed], resolved_xs=np.maximum(values[:, observed], floor)), dict(
        evaluated_layer_pairs=len(temperature), band_sample_counts=counts,
        negative_teacher_values=int(np.sum(values < 0.)),
        resolved_values_below_floor=int(np.sum(values[:, observed] < floor)),
        resolved_cross_section_floor=float(floor))


def native_chunk(plan, index, temperature, pressure, floor):
    """Intercept persistence only; never generate the original 512-row tables."""
    saved = {}
    build_tables, write_json = base.build_tables, base._write_json

    def capture(teacher, edges, temperatures, pressures, ng, directory, provenance, resolved_range=None):
        expected = base.chunk_grid(plan, index)
        if not np.array_equal(teacher.nu_grid, expected[1]) or not np.array_equal(edges, expected[3]):
            raise ValueError("Frozen loader changed the native spectral grid")
        arrays, details = fixed_teacher_values(teacher, temperature, pressure, edges, ng, resolved_range, floor)
        if "component_mapping" in plan:
            provenance["component_mapping"] = plan["component_mapping"]
            if plan["species"] == "HCN_HNC":
                database = Path(plan["mdb_path"])
                for path in (database / "direct_partition.json", database / f"{hcn_partition.STEM}.pf.original",
                             Path(hcn_partition.__file__)):
                    stat = path.stat()
                    provenance["sources"][str(path)] = dict(path=str(path), sha256=_sha256(path),
                                                            stat=[stat.st_size, stat.st_mtime_ns])
        saved.update(arrays=arrays, details=details, provenance=provenance)

    def capture_write(path, value):
        if Path(path).name != "complete.json":
            write_json(path, value)

    base.build_tables, base._write_json = capture, capture_write
    try:
        supplemental.prepare_chunk(plan, index)
    finally:
        base.build_tables, base._write_json = build_tables, write_json
    if not saved:
        raise ValueError("Frozen molecular loader did not evaluate the teacher")
    return saved


def cached_native_chunk(directory, fingerprint, plan, index, temperature, pressure, floor):
    directory.mkdir(parents=True, exist_ok=True)
    archive, completion = directory / f"chunk_{index:05d}.npz", directory / f"chunk_{index:05d}.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["fingerprint"] != fingerprint or record["sha256"] != _sha256(archive):
            raise ValueError("Changed omission-audit checkpoint; use a new output")
        base._check_sources(record["provenance"]["sources"])
        with np.load(archive, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        return arrays, record
    started = perf_counter()
    saved = native_chunk(plan, index, temperature, pressure, floor)
    temporary = archive.with_suffix(".tmp.npz")
    np.savez(temporary, **saved["arrays"])
    temporary.replace(archive)
    record = dict(fingerprint=fingerprint, sha256=_sha256(archive), chunk=index,
                  native_seconds=perf_counter()-started, **saved["details"], provenance=saved["provenance"])
    base._write_json(completion, record)
    # PreMODIT kernels have chunk-dependent shapes. Bound retained executables;
    # only the small radiation kernels below need recompilation afterwards.
    jax.clear_caches()
    gc.collect()
    return saved["arrays"], record


def mixed_depth(base_depths, added_cross_sections, added_amounts, continuum, weights):
    """Append selected absorbers without changing the original RORR order."""
    if added_cross_sections.shape[0]:
        extra = added_cross_sections * added_amounts[:, :, None, None]
        base_depths = jnp.concatenate((base_depths, extra), axis=0)
    return mix_ckd_rorr(base_depths, weights) + continuum[:, None, :]


def append_spectrum(grids, spectra, nu, fluxes):
    """Keep the preceding chunk's duplicate exactly as in the merged LBL table."""
    if grids and grids[-1][-1] > nu[0]:
        raise ValueError("Resolved chunks must be ordered without overlapping interiors")
    selection = slice(1 if grids and grids[-1][-1] == nu[0] else None, None)
    grids.append(nu[selection])
    for name, flux in fluxes.items():
        spectra[name].append(flux[selection])


def flux_at_depth(column, depth, coordinate, stellar, temperature, bottom, parameters, chemistry, *, resolved):
    """Reuse the production thermal, stellar, scattering and boundary formulas."""
    local = copy(column)
    if resolved:
        local.nu, local.star_resolved = coordinate, stellar
    else:
        local.band_nodes, local.star_bands = coordinate, stellar
        local.band_widths = jnp.ones(coordinate.shape[0])  # Shape only inside fluxes.
    local.optical_depth = lambda *args, **kwargs: depth
    return local.fluxes(temperature, bottom, parameters, chemistry, resolved=resolved)


def resolved_tables(chunk, nu, pressure):
    tables = {}
    for name, entry in chunk.items():
        _verify(entry.path, entry.sha256)
        _verify(entry.metadata_path, entry.metadata_sha256)
        opacity = OpaDiffgrid.from_saved_opa(str(entry.path))
        if not np.array_equal(opacity.nu_grid, nu):
            raise ValueError("Native and saved resolved grids differ")
        opacity.check_pressure_grid(pressure)
        info = opacity.diffgrid_info
        tables[name] = (info.inverse_temperature_grid, info.log_cross_section_grid,
                        info.log_cross_section_derivative_grid, info.log_cross_section_floor)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "observation", "state", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--species", nargs="+", choices=SPECIES, default=["H1O1", "Fe1H1"])
    parser.add_argument("--database-root", type=Path, default=Path("/home/kawahara/.database"))
    parser.add_argument("--hcn-hnc-asset", type=Path,
                        help="Isolated Harris directory prepared with the direct-state partition helper")
    parser.add_argument("--chunk-index", type=int, action="append", help="Smoke chunks only; omit for full broadband/P1")
    args = parser.parse_args(argv)
    if len(set(args.species)) != len(args.species):
        parser.error("Select each added species once")
    if args.output.resolve() in {p.resolve() for p in (args.manifest, args.observation, args.state)}:
        parser.error("The output must be separate from the inputs")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    previous = json.loads(args.output.read_text()) if args.output.exists() else None
    writable = previous is None
    report = dict(status="loading", certifies_p7=False, started_utc=datetime.now(timezone.utc).isoformat(),
                  fixed_temperature_and_chemistry=True, added_species=args.species,
                  scope="Conditional native molecular line-opacity additions; no RCE solve or global omission bound.")
    def save():
        if writable:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            base._write_json(args.output, _serializable(report))
    try:
        source = json.loads(args.state.read_text())
        state = source["state"]
        if (not source.get("success") or not all(state[key] for key in ("converged", "domain_valid", "physics_valid"))):
            raise ValueError("Use a successful converged standard forward report")
        if source.get("response_resolution") != 700.:
            raise ValueError("Use a standard forward report with the same R700 response")
        manifest, observation = load_manifest(args.manifest), load_emission_candidate(args.observation)
        if (source["asset_sha256"] != manifest.sha256 or source["configuration"] != manifest.config
                or source["observation_sha256"] != observation.sha256):
            raise ValueError("State, manifest and observation pins must match exactly")
        selected_chemical = {member for name in args.species for member in COMPONENTS[name]["chemical_species"]}
        if selected_chemical & set(manifest.config["opacity"]["ckd"]):
            raise ValueError("Audit additions must be absent from the baseline")
        column = prepare_column(manifest, observation, stream_resolved=True)
        temperature, bottom = jnp.asarray(state["temperature"]), jnp.asarray(state["bottom_temperature"])
        parameters, pressure = jnp.asarray(source["parameters"]), np.asarray(column.pressure)
        if (not np.array_equal(source["pressure_bar"], pressure)
                or not bool(column.temperature_valid(temperature, bottom, parameters))):
            raise ValueError("State pressure or physical domain differs")
        chemistry = jax.block_until_ready(jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2]))
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            raise ValueError("Invalid recomputed chemistry")
        stream = column.resolved_stream
        complete_path = manifest.paths[next(iter(column.ckd))+"_preparation"]
        complete = json.loads(complete_path.read_text())
        source_plan = complete["plan"]
        if base._sha256(Path(base.__file__)) != source_plan["settings"]["script_sha256"]:
            raise ValueError("Use the unchanged frozen molecular preparation")
        if not np.array_equal(source_plan["settings"]["pressure_grid_bar"], pressure):
            raise ValueError("This audit requires the original native pressure grid")
        workspace = args.output.with_suffix(".chunks").resolve()
        plans = {}
        for name in args.species:
            database, mapping = component_database(name, args.database_root, args.hcn_hnc_asset)
            plan = deepcopy(source_plan)
            plan.update(species=name, mdb_path=str(database), output_dir=str(workspace / "_loader"),
                        component_mapping=mapping)
            plan.pop("atomic_preparation", None)
            plan["supplemental_preparation"] = dict(driver_sha256=_sha256(Path(supplemental.__file__)),
                base_script_sha256=_sha256(Path(base.__file__)), species=name,
                default_line_list=COMPONENTS[name]["line_list"])
            database = Path(plan["mdb_path"])
            if any(not list(database.glob(pattern)) for pattern in ("*.def", "*.pf", "*.states.bz2", "*.trans.bz2")):
                raise FileNotFoundError(f"Local raw molecular assets required before this audit: {database}")
            plans[name] = plan
        codes = [Path(__file__), Path(base.__file__), Path(supplemental.__file__), Path(hcn_partition.__file__),
                 Path(__file__).with_name("rce_wasp18b_forward.py")]
        codes += sorted(Path(__file__).parent.glob("_rce*.py"))
        codes += sorted((Path(__file__).resolve().parents[1]/"src/exojax").rglob("*.py"))
        runtime = provenance()
        identity = dict(state_sha256=_sha256(args.state), asset_sha256=manifest.sha256,
                        code_sha256={str(p.resolve()): _sha256(p) for p in codes}, plans=plans,
                        runtime={key: runtime[key] for key in ("exogibbs_code_sha256", "jax", "jaxlib", "x64")},
                        added_isotope_fraction=1., native_temperature_K=np.asarray(temperature).tolist(), pressure_bar=pressure.tolist())
        fingerprint = base._digest_json(identity)
        if previous is not None and previous.get("fingerprint") != fingerprint:
            raise ValueError("Changed omission-audit report identity; use a new output")
        writable = True
        amount = jnp.diff(column.boundaries)*bar_cgs/(column.settings.gravity_cgs*m_u*chemistry.mmw[:-1])
        additions = component_fractions(args.species, chemistry, column.chemistry.species)
        amounts = {name: amount*chemistry.x[:-1, column.chemistry.species.index(name)] for name in column.ckd}
        amounts.update({name: amount*fraction for name, fraction in additions.items()})
        base_depths = jnp.stack([jax.jit(opa.xstensor_ckd)(temperature, column.pressure)
            * (amounts[name]*column.isotope_fractions[name])[:, None, None] for name, opa in column.ckd.items()])
        continuum = continuum_optical_depth(column.band_nodes.ravel(), temperature, column.pressure,
            column.boundaries, chemistry, column.chemistry.species, column.settings.gravity_cgs, column.cia)
        continuum = jnp.sum(continuum.reshape((len(pressure),)+column.band_nodes.shape)*column.band_weights, axis=-1)
        cases = {"baseline": (), **{name: (i,) for i, name in enumerate(args.species)}}
        if len(args.species) > 1:
            cases["combined"] = tuple(range(len(args.species)))
        extra_amounts = jnp.stack([amounts[name] for name in args.species])
        totals = {name: np.zeros((4, len(pressure)+1)) for name in cases}
        spectra = {name: [] for name in cases}
        grids = []
        indices = sorted(set(args.chunk_index)) if args.chunk_index is not None else list(range(source_plan["number_of_chunks"]))
        if not indices or any(i < 0 or i >= source_plan["number_of_chunks"] for i in indices):
            raise ValueError("Invalid audit chunk selection")
        lookup = {int(next(iter(chunk.values())).path.parent.name.removeprefix("chunk_")): (chunk, nu)
                  for chunk, nu in zip(stream.chunks, stream.grids)}
        incident = float(jnp.sum(irradiation_flux(column.star_bands, column.a_over_rstar, jnp.exp(parameters[2]))*column.band_widths))
        floor = float(np.exp(next(iter(column.lbl.values())).diffgrid_info.log_cross_section_floor))
        @jax.jit
        def ckd_flux(depths, extra, extra_amount, cont, nodes, star):
            depth = mixed_depth(depths, extra, extra_amount, cont, column.g_weights)
            return flux_at_depth(column, depth, nodes, star, temperature, bottom, parameters, chemistry, resolved=False)
        @jax.jit
        def resolved_flux(nu, star, tables, extra, mask):
            local = copy(column)
            local.nu, local.star_resolved = nu, star
            depth = local.optical_depth(temperature, chemistry, resolved=True, resolved_tables=tables)
            depth += jnp.sum(extra*extra_amounts[:, :, None]*mask[:, None, None], axis=0)
            return flux_at_depth(column, depth, nu, star, temperature, bottom, parameters, chemistry, resolved=True).upward[0]
        report.update(status="native_chunks", identity=identity, fingerprint=fingerprint, provenance=runtime,
            chemistry=chemistry.diagnostics._asdict(), parameters=parameters, temperature_K=temperature,
            bottom_temperature_K=bottom, pressure_bar=pressure, selected_chunks=indices,
            added_number_fractions=additions,
            component_mapping={name: plans[name]["component_mapping"] for name in args.species},
            full_broadband=len(indices)==source_plan["number_of_chunks"], chunks=[],
            limitations=["Fixed T/composition only; significant heating changes require RCE re-equilibration.",
                "Each component uses one isotopologue with fraction 1 and native H2/default broadening.",
                "Finite native line-list extent; missing electronic bands, continua and unselected absorbers remain outside this audit.",
                "Unchanged CKD/RORR and finite teacher sampling approximations; no new global error bound."])
        if "HCN_HNC" in args.species:
            report["hcn_hnc_normalization"] = hcn_partition.normalization_diagnostics(
                plans["HCN_HNC"]["mdb_path"], np.asarray(temperature),
                np.asarray(chemistry.x[:-1, column.chemistry.species.index("C1H1N1_hcn")]),
                np.asarray(chemistry.x[:-1, column.chemistry.species.index("C1H1N1_hnc")]))
            report["limitations"].extend(hcn_partition.ASSUMPTIONS)
        save()
        for index in indices:
            print("Auditing molecular additions chunk", index+1, "/", source_plan["number_of_chunks"], flush=True)
            native, records = [], []
            for name in args.species:
                arrays, record = cached_native_chunk(workspace/name, fingerprint, plans[name], index,
                                                     np.asarray(temperature), pressure, floor)
                native.append(arrays)
                records.append(dict(species=name, sha256=record["sha256"], native_seconds=record["native_seconds"],
                    number_of_lines=record["provenance"]["number_of_lines"],
                    no_lines_reason=record["provenance"].get("no_lines_reason")))
            first = index*source_plan["settings"]["bands_per_chunk"]
            edges = native[0]["edges"]
            selection = slice(first, first+len(edges))
            if any(not np.array_equal(a["edges"], np.asarray(column.band_edges[selection])) for a in native):
                raise ValueError("Native and baseline CKD band grids differ")
            extra = jnp.asarray(np.stack([np.exp(a["logk"]) for a in native]))
            for name, selected in cases.items():
                take = jnp.asarray(selected, dtype=int)
                flux = np.asarray(ckd_flux(base_depths[..., selection], extra[take], extra_amounts[take],
                    continuum[:, selection], column.band_nodes[selection], column.star_bands[selection]))
                if not np.all(np.isfinite(flux)):
                    raise ValueError("Nonfinite CKD boundary flux")
                totals[name] += np.einsum("flgb,g,b->fl", flux, column.g_weights, column.band_widths[selection])
            if index in lookup:
                chunk, nu = lookup[index]
                if any(not np.array_equal(a["resolved_nu"], nu) for a in native):
                    raise ValueError("Native addition and baseline resolved grids differ")
                tables = resolved_tables(chunk, nu, pressure)
                added = jnp.asarray(np.stack([a["resolved_xs"] for a in native]))
                top_fluxes = {}
                for name, selected in cases.items():
                    mask = np.isin(np.arange(len(args.species)), selected).astype(float)
                    flux = np.asarray(resolved_flux(jnp.asarray(nu), jnp.asarray(column.stellar.at(nu)), tables, added, jnp.asarray(mask)))
                    if not np.all(np.isfinite(flux)):
                        raise ValueError("Nonfinite resolved emergent flux")
                    top_fluxes[name] = flux
                append_spectrum(grids, spectra, nu, top_fluxes)
                del tables
            report["chunks"].append(dict(index=index, species=records))
            report["boundary_flux_cgs"] = {name: dict(zip(Fluxes._fields, value)) for name, value in totals.items()}
            save()
        comparisons = {}
        if report["full_broadband"]:
            report["baseline_boundary_reproduction_cgs"] = dict(zip(Fluxes._fields,
                np.max(abs(totals["baseline"] - np.array([source["broadband_flux"][name]
                    for name in Fluxes._fields])), axis=1)))
        for name in cases:
            if name == "baseline":
                continue
            delta = totals[name]-totals["baseline"]
            comparisons[name] = dict(boundary_difference_cgs=dict(zip(Fluxes._fields, delta)),
                max_boundary_difference_over_incident=dict(zip(Fluxes._fields, np.max(abs(delta), axis=1)/incident)),
                layer_heating_difference_cgs=np.diff(delta[-1]),
                max_layer_heating_difference_over_incident=float(np.max(abs(np.diff(delta[-1]))))/incident)
        if grids and np.array_equal(np.concatenate(grids), stream.nu):
            predictions = {name: np.asarray(stream.prediction(column, np.concatenate(values), 700.)) for name, values in spectra.items()}
            report["resolved_prediction_R700"] = predictions
            report["baseline_prediction_reproduction"] = spectral_error(predictions["baseline"], source["prediction"], observation.uncertainty)
            for name in comparisons:
                comparisons[name]["resolved_P1_difference"] = spectral_error(predictions[name], predictions["baseline"], observation.uncertainty)
        else:
            report["resolved_prediction_scope"] = "Partial observed coverage; no full P1 comparison is valid"
        report.update(status="complete_diagnostic_only", comparisons=comparisons,
            incident_flux_for_normalization_cgs=incident, device_memory_stats=jax.devices()[0].memory_stats())
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report.update(seconds=perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
