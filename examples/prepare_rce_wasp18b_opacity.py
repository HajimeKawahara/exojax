"""Download ExoMol lines and build matched, resumable WASP-18b opacity tables.

Run one species per invocation; spectral chunks run in fresh child processes.
The resulting manifest is a preparation inventory, not a physical acceptance.
No download or directory creation occurs with --dry-run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from _rce_wasp18b_assets import SCHEMA_VERSION, _sha256


LINE_LISTS = {
    "H2O1": "H2O/1H2-16O/POKAZATEL",
    "C1O1": "CO/12C-16O/Li2015",
    "C1O2": "CO2/12C-16O2/UCL-4000",
    "H3N1": "NH3/14N-1H3/CoYuTe",
    "C1H4": "CH4/12C-1H4/YT34to10",
}
CORE_SPECIES = ("H2O1", "C1O1", "C1O2")


def parser_with_defaults():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database-root", type=Path, default=Path("/home/kawahara/.database"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--species", choices=sorted(LINE_LISTS), default="H2O1")
    parser.add_argument("--mdb-path", type=Path, help="Override the selected species line-list directory.")
    parser.add_argument("--nu-min", type=float, default=200.0)
    parser.add_argument("--nu-max", type=float, default=50000.0)
    parser.add_argument("--resolved-nu-min", type=float, default=3300.0)
    parser.add_argument("--resolved-nu-max", type=float, default=12500.0)
    parser.add_argument("--ckd-resolution", type=float, default=1000.0)
    parser.add_argument("--samples-per-band", type=int, default=512)
    parser.add_argument("--bands-per-chunk", type=int, default=16)
    parser.add_argument("--wing-padding", type=float, default=25.0, help="Teacher grid halo in cm-1; convergence must be tested.")
    parser.add_argument("--temperature-min", type=float, default=1500.0)
    parser.add_argument("--temperature-max", type=float, default=4500.0)
    parser.add_argument("--temperature-nodes", type=int, default=16)
    parser.add_argument("--ng", type=int, default=16)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--assemble-species", nargs="+", choices=sorted(LINE_LISTS), default=list(CORE_SPECIES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, help=argparse.SUPPRESS)
    return parser


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_plan(args):
    """Validate configuration and define identical grids for every species."""
    root = args.database_root.expanduser().resolve()
    output = (args.output_dir or root / "rce_wasp18b/prepared").expanduser().resolve()
    inventory = (args.inventory or root / "rce_wasp18b/local_inventory.json").expanduser().resolve()
    config = json.loads(inventory.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported input inventory schema")
    pressure = np.asarray(config["column"]["pressure_boundaries_bar"], dtype=float)
    if (pressure.ndim != 1 or pressure.size < 3 or not np.all(np.isfinite(pressure))
            or np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0)):
        raise ValueError("Pressure boundaries must be positive and increasing")
    for name in ("nu_min", "nu_max", "resolved_nu_min", "resolved_nu_max", "ckd_resolution", "temperature_min", "temperature_max", "wing_padding"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not args.nu_min <= args.resolved_nu_min < args.resolved_nu_max <= args.nu_max:
        raise ValueError("Resolved range must lie within the broadband grid")
    if args.nu_min >= args.nu_max or args.temperature_min >= args.temperature_max:
        raise ValueError("Spectral and temperature minima must precede maxima")
    for name in ("samples_per_band", "bands_per_chunk", "temperature_nodes", "ng"):
        if getattr(args, name) < 2:
            raise ValueError(f"{name} must be at least two")
    if args.wing_padding >= args.nu_min:
        raise ValueError("wing_padding must be smaller than nu_min")
    nband = int(math.ceil(math.log(args.nu_max / args.nu_min) * args.ckd_resolution))
    settings = {name: getattr(args, name) for name in (
        "nu_min", "nu_max", "resolved_nu_min", "resolved_nu_max", "ckd_resolution", "samples_per_band", "bands_per_chunk",
        "wing_padding", "temperature_min", "temperature_max", "temperature_nodes", "ng",
    )}
    settings.update(
        pressure_grid_bar=np.sqrt(pressure[:-1] * pressure[1:]).tolist(),
        pressure_boundaries_bar=pressure.tolist(), number_of_bands=nband,
        temperature_grid_K=(1.0 / np.linspace(1.0 / args.temperature_max,
                                            1.0 / args.temperature_min,
                                            args.temperature_nodes)).tolist(),
        profile_kernel="real_space", broadener="H2", line_strength_cutoff=0.0,
        broadening_resolution=0.2, premodit_diffmode=0, teacher_layers_per_batch=4,
        script_sha256=_sha256(Path(__file__).resolve()),
    )
    step = math.log(args.nu_max / args.nu_min) / (nband * args.samples_per_band)
    first = int(math.ceil(math.log(args.resolved_nu_min / args.nu_min) / step))
    last = int(math.floor(math.log(args.resolved_nu_max / args.nu_min) / step))
    nnu = last - first + 1
    return dict(
        settings=settings, grid_fingerprint=_digest_json(settings),
        database_root=str(root), output_dir=str(output), inventory=str(inventory),
        species=args.species, mdb_path=str((args.mdb_path or root / LINE_LISTS[args.species]).expanduser().resolve()),
        number_of_chunks=math.ceil(nband / args.bands_per_chunk),
        resolved_points=nnu,
        broadband_teacher_points=nband * args.samples_per_band + 1,
        diffgrid_uncompressed_bytes_per_species=2 * (pressure.size - 1) * args.temperature_nodes * nnu * 8,
        ckd_uncompressed_bytes_per_species=(pressure.size - 1) * args.temperature_nodes * args.ng * nband * 8,
        status="Preparation only; resolution, wings, line completeness and omitted species need convergence/audit.",
    )


def chunk_grid(plan, index):
    """Return a globally aligned core and a padded, even-length ESLOG grid."""
    spec = plan["settings"]
    if not 0 <= index < plan["number_of_chunks"]:
        raise ValueError("Chunk index outside the configured grid")
    first = index * spec["bands_per_chunk"]
    last = min(first + spec["bands_per_chunk"], spec["number_of_bands"])
    step = math.log(spec["nu_max"] / spec["nu_min"]) / (spec["number_of_bands"] * spec["samples_per_band"])
    positions = np.arange(spec["number_of_bands"] + 1) * spec["samples_per_band"]
    all_edges = spec["nu_min"] * np.exp(step * positions)
    # Preserve exact requested endpoints for stellar coverage checks.
    all_edges[0], all_edges[-1] = spec["nu_min"], spec["nu_max"]
    edges = np.column_stack((all_edges[first:last], all_edges[first + 1:last + 1]))
    core_first, core_last = first * spec["samples_per_band"], last * spec["samples_per_band"]
    low = int(math.floor(math.log((edges[0, 0] - spec["wing_padding"]) / spec["nu_min"]) / step))
    high = int(math.ceil(math.log((edges[-1, 1] + spec["wing_padding"]) / spec["nu_min"]) / step))
    high += (high - low + 1) % 2
    teacher_nu = spec["nu_min"] * np.exp(step * np.arange(low, high + 1))
    core = slice(core_first - low, core_last - low + 1)
    core_nu = teacher_nu[core].copy()
    core_nu[0], core_nu[-1] = edges[0, 0], edges[-1, 1]
    return teacher_nu, core_nu, core, edges


class CroppedTeacher:
    """Apply the same finite spectral halo to the CKD and Diffgrid teacher."""

    def __init__(self, teacher, nu_grid, selection, provenance):
        self.teacher = teacher
        self.nu_grid = nu_grid
        self.selection = selection
        self.method = getattr(teacher, "method", "zero_line_opacity")
        self.ready = True
        self.provenance = provenance
        if teacher is not None:
            self.molmass = getattr(teacher, "molmass", None)

    def meta(self):
        return self.provenance

    def xsmatrix(self, temperature, pressure):
        import jax.numpy as jnp

        if self.teacher is None:
            return jnp.zeros((len(temperature), len(self.nu_grid)), dtype=temperature.dtype)
        import jax

        chunks = []
        for start in range(0, len(temperature), 4):
            result = self.teacher.xsmatrix(temperature[start:start + 4], pressure[start:start + 4])[:, self.selection]
            if not isinstance(result, jax.core.Tracer):
                result.block_until_ready()
            chunks.append(result)
        return jnp.concatenate(chunks, axis=0)


def build_tables(teacher, edges, temperatures, pressure, ng, directory, provenance, resolved_range=None):
    """Use the existing builders and persistence formats for one spectral chunk."""
    import jax.numpy as jnp
    from exojax.opacity import OpaCKD, OpaDiffgrid, saveopa

    ckd = OpaCKD(teacher, Ng=ng, band_width=float(np.median(np.diff(edges, axis=1))), band_spacing="log")
    ckd.band_edges = jnp.asarray(edges)
    ckd.nu_bands = jnp.sqrt(ckd.band_edges[:, 0] * ckd.band_edges[:, 1])
    ckd.precompute_tables(np.sort(temperatures), pressure)
    ckd.save_tables(str(directory / "ckd.npz"), overwrite=True)
    resolved_teacher = teacher
    if resolved_range is not None:
        indices = np.flatnonzero((teacher.nu_grid >= resolved_range[0]) & (teacher.nu_grid <= resolved_range[1]))
        if not indices.size:
            return
        crop = slice(int(indices[0]), int(indices[-1]) + 1)
        resolved_teacher = CroppedTeacher(teacher, teacher.nu_grid[crop], crop, provenance)
    resolved = OpaDiffgrid(resolved_teacher, temperatures, pressure, profile_kernel=None)
    resolved.teacher_profile_kernel = provenance.get("profile_kernel")
    saveopa(resolved, str(directory / "diffgrid.npz"), format="npz", extra_meta=provenance)


def _file_entry(path, source):
    return dict(path=str(path.resolve()), sha256=_sha256(path), source=source)


def _check_files(files):
    for item in files.values():
        path = Path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"Changed or missing preparation artifact: {path}")


def _source_files(mdb, database, cache_path):
    """Hash the actual raw files and parser caches, caching unchanged file hashes."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    candidates = set()
    manager = mdb.get_datafile_manager() if mdb is not None else None
    for name in ("trans_file", "states_file", "pf_file", "def_file", "broad_file"):
        value = getattr(mdb, name, ())
        for path in value if isinstance(value, (list, tuple)) else (value,):
            if not path:
                continue
            path = Path(path)
            candidates.add(path)
            if manager is not None and name in ("trans_file", "states_file"):
                candidates.add(Path(manager.cache_file(path)))
    # Include metadata even for zero-line chunks beyond a declared line-list end.
    for pattern in ("*.def", "*.pf", "*.broad"):
        candidates.update(database.glob(pattern))
    files = {}
    for path in sorted(candidates):
        if not path.is_file():
            continue
        path = path.resolve()
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        previous = cache.get(str(path), {})
        digest = previous.get("sha256") if previous.get("stat") == stamp else _sha256(path)
        cache[str(path)] = dict(sha256=digest, stat=stamp)
        files[str(path)] = dict(path=str(path), sha256=digest, stat=stamp)
    _write_json(cache_path, cache)
    return files


def _check_sources(files):
    for item in files.values():
        path = Path(item["path"])
        if not path.is_file():
            raise ValueError(f"Missing source from resumed preparation: {path}")
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != item["stat"] and _sha256(path) != item["sha256"]:
            raise ValueError(f"Changed source from resumed preparation: {path}; use a new output directory")


def _declared_maximum(database):
    for path in database.glob("*.def"):
        for line in path.read_text().splitlines():
            if "Maximum wavenumber (in cm-1)" in line:
                return float(line.split("#")[0].split()[0])
    return None


def prepare_chunk(plan, index):
    import jax
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit

    jax.config.update("jax_enable_x64", True)
    directory = Path(plan["output_dir"]) / plan["species"] / f"chunk_{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    grid, core_nu, selection, edges = chunk_grid(plan, index)
    database = Path(plan["mdb_path"])
    spec = plan["settings"]
    maximum = _declared_maximum(database)
    mdb, teacher = None, None
    no_lines_reason = None
    if maximum is not None and maximum > 0 and grid[0] >= maximum:
        no_lines_reason = "Outside the line-list maximum wavenumber declared in its .def; zero line opacity assumed."
    else:
        mdb = MdbExomol(
            str(database), nurange=[float(grid[0]), float(grid[-1])],
            local_databases=".", gpu_transfer=False, activation=False,
            Ttyp=spec["temperature_max"], crit=0.0, bkgdatm="H2",
            broadf=True, broadf_download=True,
        )
        if np.min(mdb.T_gQT) > spec["temperature_min"] or np.max(mdb.T_gQT) < spec["temperature_max"]:
            raise ValueError("Partition-function table does not cover the requested temperatures")
        if np.any(mdb.df_load_mask):
            mdb.activate(mdb.df)
            del mdb.df
            teacher = OpaPremodit(
                mdb, grid, auto_trange=(spec["temperature_min"], spec["temperature_max"]),
                diffmode=spec["premodit_diffmode"], profile_kernel=spec["profile_kernel"],
                broadening_resolution={"mode": "manual", "value": spec["broadening_resolution"]},
            )
        else:
            no_lines_reason = "No line centers within the padded chunk; zero line opacity assumed."
    sources = _source_files(mdb, database, directory.parent / "source_hashes.json")
    provenance = dict(
        grid_fingerprint=plan["grid_fingerprint"], species=plan["species"],
        line_list=plan["mdb_path"], sources=sources, profile_kernel=spec["profile_kernel"],
        broadener="H2; ExoMol .broad when available, otherwise .def defaults",
        no_lines_reason=no_lines_reason, number_of_lines=0 if teacher is None else int(len(mdb.nu_lines)),
        wing_padding_cm1=spec["wing_padding"], declared_maximum_wavenumber_cm1=_declared_maximum(database),
        line_completeness="Not established by table temperature extent.",
    )
    teacher = CroppedTeacher(teacher, core_nu, selection, provenance)
    build_tables(teacher, edges, np.asarray(spec["temperature_grid_K"]),
                 np.asarray(spec["pressure_grid_bar"]), spec["ng"], directory, provenance,
                 (spec["resolved_nu_min"], spec["resolved_nu_max"]))
    files = {name: _file_entry(directory / name, "ExoJAX matched PreMODIT preparation")
             for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json") if (directory / name).exists()}
    _write_json(directory / "complete.json", dict(
        grid_fingerprint=plan["grid_fingerprint"], mdb_path=plan["mdb_path"],
        files=files, provenance=provenance,
    ))


def completed_chunks(plan):
    records = []
    for index in range(plan["number_of_chunks"]):
        path = Path(plan["output_dir"]) / plan["species"] / f"chunk_{index:05d}" / "complete.json"
        if not path.exists():
            records.append(None)
            continue
        record = json.loads(path.read_text())
        if record["grid_fingerprint"] != plan["grid_fingerprint"] or record["mdb_path"] != plan["mdb_path"]:
            raise ValueError("Existing preparation uses different settings; choose a new output directory")
        _check_files(record["files"])
        _check_sources(record["provenance"]["sources"])
        records.append(record)
    return records


def merge_species(plan, records):
    """Join chunk tables on the host, preserving the standard archive contracts."""
    import jax
    from exojax.opacity import OpaCKD, OpaDiffgrid, saveopa
    from exojax.opacity.diffgrid.contracts import DiffgridInfo

    jax.config.update("jax_enable_x64", True)
    directory = Path(plan["output_dir"]) / plan["species"]
    ckd = OpaCKD.load_only().load_tables(records[0]["files"]["ckd.npz"]["path"])
    fields = {name: [] for name in ("log_kggrid", "nu_bands", "band_edges")}
    nu, values, slopes = [], [], []
    first_resolved = None
    for index, record in enumerate(records):
        with np.load(record["files"]["ckd.npz"]["path"], allow_pickle=False) as data:
            for name in fields:
                fields[name].append(np.asarray(data[name]))
        if "diffgrid.npz" not in record["files"]:
            continue
        with np.load(record["files"]["diffgrid.npz"]["path"], allow_pickle=False) as data:
            if first_resolved is None:
                first_resolved = {name: np.asarray(data[name]) for name in (
                    "pressure_grid", "temperature_grid", "inverse_temperature_grid", "log_cross_section_floor")}
            # Consecutive core grids share exactly one endpoint.
            crop = slice(1 if nu and nu[-1][-1] == data["nu_grid"][0] else None, None)
            nu.append(np.asarray(data["nu_grid"])[crop])
            values.append(np.asarray(data["log_cross_section_grid"])[..., crop])
            slopes.append(np.asarray(data["log_cross_section_derivative_grid"])[..., crop])
    if first_resolved is None:
        raise ValueError("No resolved samples; widen resolved range or increase spectral sampling")
    fields = {name: np.concatenate(chunks, axis=-1 if name == "log_kggrid" else 0)
              for name, chunks in fields.items()}
    ckd.ckd_info = replace(ckd.ckd_info, **fields)
    ckd.nu_bands, ckd.band_edges = fields["nu_bands"], fields["band_edges"]
    ckd._expected_base_meta = dict(grid_fingerprint=plan["grid_fingerprint"],
                                   line_list=plan["mdb_path"], chunks=[record["files"]["ckd.npz"]["sha256"] for record in records])
    ckd.save_tables(str(directory / "ckd.npz"), overwrite=True)
    resolved = OpaDiffgrid.__new__(OpaDiffgrid)
    resolved.nu_grid = np.concatenate(nu)
    resolved.method, resolved.teacher_method = "diffgrid", "premodit"
    resolved.teacher_profile_kernel = plan["settings"]["profile_kernel"]
    resolved._apply_diffgrid_info(DiffgridInfo(
        **first_resolved, log_cross_section_grid=np.concatenate(values, axis=-1),
        log_cross_section_derivative_grid=np.concatenate(slopes, axis=-1)))
    resolved.ready = True
    saveopa(resolved, str(directory / "diffgrid.npz"), format="npz", extra_meta=plan)
    files = {name: _file_entry(directory / name, "ExoJAX matched PreMODIT preparation")
             for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json")}
    _write_json(directory / "complete.json", dict(
        plan=plan, files=files, no_line_chunks=[i for i, rec in enumerate(records) if rec["provenance"]["no_lines_reason"]],
        chunk_records=[_file_entry(directory / f"chunk_{i:05d}" / "complete.json", "Preparation provenance")
                       for i in range(len(records))],
    ))


def assemble_manifest(plan, species):
    """Replace the inventory opacity mapping with explicitly prepared species."""
    from _rce_wasp18b_assets import load_manifest

    inventory = load_manifest(plan["inventory"])
    config = inventory.config
    config["files"] = {name: dict(item, path=str(inventory.paths[name])) for name, item in config["files"].items()}
    old_species = set(config["opacity"]["ckd"])
    config["opacity"]["ckd"], config["opacity"]["lbl"] = {}, {}
    config["opacity"]["mapping"] = {name: config["opacity"]["mapping"][name] for name in species}
    preparations = {}
    for name in species:
        record_path = Path(plan["output_dir"]) / name / "complete.json"
        record = json.loads(record_path.read_text())
        if record["plan"]["grid_fingerprint"] != plan["grid_fingerprint"]:
            raise ValueError("Species use different preparation grids/settings")
        _check_files(record["files"])
        _check_files({str(i): entry for i, entry in enumerate(record["chunk_records"])})
        for key, filename in (("ckd", "ckd.npz"), ("lbl", "diffgrid.npz"), ("lbl_meta", "diffgrid_metadata.json")):
            config["files"][f"{name}_{key}"] = record["files"][filename]
        config["files"][f"{name}_preparation"] = _file_entry(record_path, "ExoJAX preparation provenance")
        config["opacity"]["ckd"][name] = dict(asset=f"{name}_ckd", format="npz")
        config["opacity"]["lbl"][name] = dict(asset=f"{name}_lbl", format="diffgrid", metadata_asset=f"{name}_lbl_meta")
        config["opacity"]["mapping"][name]["broadening"] = "Fixed H2 broadening; ExoMol .broad or .def fallback; He/H dependence unaudited."
        source_record = json.loads(Path(record["chunk_records"][0]["path"]).read_text())
        preparations[name] = dict(
            line_list=record["plan"]["mdb_path"], no_line_chunks=record["no_line_chunks"],
            declared_maximum_wavenumber_cm1=source_record["provenance"].get("declared_maximum_wavenumber_cm1"),
            zero_line_policy="Zero line opacity outside declared line-list extent or empty padded intervals; missing transitions are not certified absent.",
            resolved_coverage_cm1=[plan["settings"]["resolved_nu_min"], plan["settings"]["resolved_nu_max"]],
            line_completeness="Temperature nodes do not certify the completeness of this line list.",
        )
    omitted = config["opacity"].setdefault("omitted_absorbers", [])
    omitted.extend(sorted(old_species - set(species) - set(omitted)))
    config["readiness"] = "Matched molecular tables prepared; stellar/CIA coverage, omitted absorbers and convergence remain unaudited."
    config["missing_requirements"] = [
        "Stellar surface spectrum covering the complete broadband grid.",
        "CIA temperature coverage and applicable pressure/composition domain.",
        "Line-list completeness at the requested temperatures; isotope/broadening and omitted absorbers audit.",
        "Spectral resolution, wing padding, temperature nodes and CKD quadrature convergence.",
    ]
    config["preparation"] = dict(settings=plan["settings"], grid_fingerprint=plan["grid_fingerprint"], species=preparations)
    path = Path(plan["output_dir"]) / "prepared_manifest.json"
    _write_json(path, config)
    return path


def main(argv=None):
    parser = parser_with_defaults()
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.assemble:
            print(assemble_manifest(plan, args.assemble_species))
            return 0
        if args.chunk_index is not None:
            prepare_chunk(plan, args.chunk_index)
            return 0
        records = completed_chunks(plan)
        complete_path = Path(plan["output_dir"]) / plan["species"] / "complete.json"
        if complete_path.exists() and all(record is not None for record in records):
            complete = json.loads(complete_path.read_text())
            if complete["plan"]["grid_fingerprint"] != plan["grid_fingerprint"]:
                raise ValueError("Completed species uses different settings; choose a new output directory")
            _check_files(complete["files"])
            print(f"Reusing completed {args.species} tables", flush=True)
            return 0
        arguments = list(sys.argv[1:] if argv is None else argv)
        for index, record in enumerate(records):
            if record is None:
                print(f"Preparing {args.species} chunk {index + 1}/{len(records)}", flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments, "--chunk-index", str(index)], check=True)
            else:
                print(f"Reusing {args.species} chunk {index + 1}/{len(records)}", flush=True)
        # No scientific import occurs in this parent before the host-side merge.
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        merge_species(plan, completed_chunks(plan))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
