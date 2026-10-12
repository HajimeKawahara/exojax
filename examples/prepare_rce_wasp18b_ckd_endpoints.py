"""Append CKD endpoint bands while preserving the original P7 tables exactly.

The original logarithmic teacher spacing is extended by 693 bands at each end
for the standard P7 grid. Lower, upper and combined variants share the same
new endpoint coefficients and reuse the original resolved opacity archives.
This checks numerical endpoint sensitivity within the stated opacity model;
it does not establish completeness of ultraviolet absorption.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_rce_wasp18b_ckd_refinement as refinement

base = refinement.base
VARIANTS = {"lower": ("lower",), "upper": ("upper",), "both": ("lower", "upper")}
LIMITATIONS = [
    "Numerical endpoint sensitivity only; ultraviolet absorption completeness is not certified.",
    "Existing H-minus bound-free is zero at wavelength <= 0.125 um; free-free is zero below 0.1823 um.",
    "Existing molecular line-list upper limits imply zero line opacity beyond their declared limits, not physical absence.",
    "CIA remains zero outside its tabulated spectral interval; atomic bound-free and ionized line opacity remain omitted.",
]


def make_plan(args, *, extension_bands=None):
    """Keep the original teacher spacing and remain inside the 100 cm-1 tail."""
    anchor = refinement.make_plan(args)
    original = anchor["source_plan"]
    settings = original["settings"]
    step = math.log(settings["nu_max"] / settings["nu_min"]) / settings["number_of_bands"]
    count = math.floor(math.log(2.) / step) if extension_bands is None else extension_bands
    if not isinstance(count, int) or count < 1:
        raise ValueError("Endpoint extension requires a positive integer band count")
    lower = settings["nu_min"] * math.exp(-count * step)
    upper = settings["nu_max"] * math.exp(count * step)
    if lower < 100. or lower <= settings["wing_padding"]:
        raise ValueError("Lower endpoint must remain >= 100 cm-1 and above the teacher halo width")
    identity = dict(source_plan=original, base_preparation=anchor["base_preparation"],
                    extension_bands=count, band_log_step=step,
                    driver_sha256=base._sha256(Path(__file__)),
                    refinement_driver_sha256=base._sha256(Path(refinement.__file__)))
    fingerprint = base._digest_json(identity)
    bytes_per_band = (len(settings["temperature_grid_K"]) * len(settings["pressure_grid_bar"])
                      * settings["ng"] * 8)
    merged_bytes = {name: bytes_per_band * (settings["number_of_bands"] + len(ends) * count)
                    for name, ends in VARIANTS.items()}
    return dict(
        schema_version="exojax.rce-wasp18b-ckd-endpoints/1", identity=identity,
        fingerprint=fingerprint, source_plan=original,
        source_preparation_path=anchor["source_preparation_path"],
        source_merge_ready=anchor["source_merge_ready"], species=original["species"],
        output_dir=anchor["output_dir"], added_bands_per_end=count,
        chunks_per_end=math.ceil(count / settings["bands_per_chunk"]),
        original_bands=settings["number_of_bands"],
        original_teacher_grid_fingerprint=original["grid_fingerprint"],
        required_stellar_coverage_cm1=[lower, upper],
        variants={name: dict(ends=list(ends), grid_fingerprint=base._digest_json(dict(
            original_grid=original["grid_fingerprint"], count=count, ends=list(ends))))
                  for name, ends in VARIANTS.items()},
        limitations=LIMITATIONS,
        storage_estimate=dict(merged_coefficient_bytes=merged_bytes,
            chunks_and_three_merged_coefficient_bytes=sum(merged_bytes.values()) + 2 * count * bytes_per_band,
            repeat_merge_peak_coefficient_bytes=sum(merged_bytes.values()) + 2 * count * bytes_per_band + max(merged_bytes.values()),
            note="Uncompressed coefficients; excludes small grids/metadata, new raw line caches and filesystem overhead."),
        declared_line_list_maximum_cm1=(None if original.get("atomic_preparation") else
                                       base._declared_maximum(Path(original["mdb_path"]))),
    )


def endpoint_grid(plan, end, index):
    """Extend signed original band/sample indices, with exact shared endpoints."""
    if end not in ("lower", "upper") or not 0 <= index < plan["chunks_per_end"]:
        raise ValueError("Invalid endpoint or chunk index")
    spec = plan["source_plan"]["settings"]
    count, nband = plan["added_bands_per_end"], spec["number_of_bands"]
    offset = -count if end == "lower" else nband
    first = offset + index * spec["bands_per_chunk"]
    last = offset + min((index + 1) * spec["bands_per_chunk"], count)
    step = plan["identity"]["band_log_step"] / spec["samples_per_band"]
    indices = np.arange(first, last + 1)
    points = spec["nu_min"] * np.exp(step * (indices * spec["samples_per_band"]))
    points[indices == 0], points[indices == nband] = spec["nu_min"], spec["nu_max"]
    edges = np.column_stack((points[:-1], points[1:]))
    low = math.floor(math.log((points[0] - spec["wing_padding"]) / spec["nu_min"]) / step)
    high = math.ceil(math.log((points[-1] + spec["wing_padding"]) / spec["nu_min"]) / step)
    high += (high - low + 1) % 2
    padded = spec["nu_min"] * np.exp(step * np.arange(low, high + 1))
    selection = slice(first * spec["samples_per_band"] - low,
                      last * spec["samples_per_band"] - low + 1)
    core = padded[selection].copy()
    core[0], core[-1] = points[0], points[-1]
    return padded, core, selection, edges


def _directory(plan, end, index):
    return Path(plan["output_dir"]) / end / plan["species"] / f"chunk_{index:05d}"


def _source_records(plan):
    """Verify original provenance and collect hashes for overlapping inputs."""
    path = Path(plan["source_preparation_path"])
    base._check_files({"common": plan["identity"]["base_preparation"]})
    source = json.loads(path.read_text())
    if source["plan"] != plan["source_plan"]:
        raise ValueError("Original preparation plan changed")
    hashes = {}
    for item in source["chunk_records"]:
        base._check_files({"chunk": item})
        provenance = json.loads(Path(item["path"]).read_text())["provenance"]
        for key, value in provenance["sources"].items():
            if key in hashes and hashes[key] != value["sha256"]:
                raise ValueError("Original preparation contains inconsistent source hashes")
            hashes[key] = value["sha256"]
    return source, hashes


def prepare_chunk(plan, end, index):
    """Reuse frozen line loaders, replacing only their grid and table writer."""
    import jax
    import jax.numpy as jnp
    from exojax.opacity import OpaCKD

    jax.config.update("jax_enable_x64", True)
    _, known_sources = _source_records(plan)
    expected = endpoint_grid(plan, end, index)
    original = deepcopy(plan["source_plan"])
    original["output_dir"] = str(Path(plan["output_dir"]) / end)
    driver, _ = refinement._driver(plan["species"])
    chunk_grid, build_tables, write_json = base.chunk_grid, base.build_tables, base._write_json
    saved = {}

    def grid(_plan, _index):
        if _index != index:
            raise ValueError("Frozen loader requested a different endpoint chunk")
        return expected

    def build(teacher, edges, temperatures, pressure, ng, directory, provenance, resolved_range=None):
        settings = plan["source_plan"]["settings"]
        if (not np.array_equal(teacher.nu_grid, expected[1]) or not np.array_equal(edges, expected[3])
                or not np.array_equal(temperatures, settings["temperature_grid_K"])
                or not np.array_equal(pressure, settings["pressure_grid_bar"]) or ng != settings["ng"]):
            raise ValueError("Endpoint teacher coordinates or quadrature changed")
        for key, item in provenance["sources"].items():
            if key in known_sources and item["sha256"] != known_sources[key]:
                raise ValueError("A source shared with the original preparation changed")
        provenance["anchor_grid_fingerprint"] = provenance.pop("grid_fingerprint", original["grid_fingerprint"])
        provenance["grid_fingerprint"] = plan["variants"][end]["grid_fingerprint"]
        provenance["endpoint_extension"] = dict(end=end, limitations=plan["limitations"],
                                                teacher_nu_sha256=refinement._array_hash(teacher.nu_grid))
        ckd = OpaCKD(teacher, Ng=ng, band_width=float(np.median(np.diff(edges, axis=1))), band_spacing="log")
        ckd.band_edges = jnp.asarray(edges)
        ckd.nu_bands = jnp.sqrt(ckd.band_edges[:, 0] * ckd.band_edges[:, 1])
        ckd.precompute_tables(np.sort(temperatures), pressure)
        ckd.base_opa = None
        ckd._expected_base_meta = dict(teacher=provenance, endpoint_fingerprint=plan["fingerprint"],
                                      band_edges_sha256=refinement._array_hash(edges))
        path = _directory(plan, end, index) / "ckd.npz"
        refinement._save_ckd(ckd, path)
        saved.update(files={"ckd.npz": base._file_entry(path, "Additional endpoint CKD only")},
                     provenance=provenance)

    def suppress_original_complete(path, value):
        if Path(path).name != "complete.json":
            write_json(path, value)

    base.chunk_grid, base.build_tables, base._write_json = grid, build, suppress_original_complete
    try:
        driver.prepare_chunk(original, index)
    finally:
        base.chunk_grid, base.build_tables, base._write_json = chunk_grid, build_tables, write_json
    record = dict(fingerprint=plan["fingerprint"], end=end, index=index,
                  original_preparation=base._file_entry(Path(plan["source_preparation_path"]), "Original interior bands and LBL"), **saved)
    base._write_json(_directory(plan, end, index) / "complete.json", record)
    return record


def completed_chunks(plan, end):
    records = []
    for index in range(plan["chunks_per_end"]):
        path = _directory(plan, end, index) / "complete.json"
        if not path.exists():
            records.append(None)
            continue
        record = json.loads(path.read_text())
        if record["fingerprint"] != plan["fingerprint"] or record["end"] != end or record["index"] != index:
            raise ValueError("Existing endpoint preparation has different settings")
        base._check_files(record["files"])
        base._check_files({"original": record["original_preparation"]})
        base._check_sources(record["provenance"]["sources"])
        records.append(record)
    return records


def merge_variant(plan, variant_name):
    """Copy every original coefficient and band edge unchanged into one variant."""
    import jax
    from exojax.opacity import OpaCKD

    jax.config.update("jax_enable_x64", True)
    source, _ = _source_records(plan)
    base._check_files(source["files"])
    ckd = OpaCKD.load_only().load_tables(source["files"]["ckd.npz"]["path"])
    fields = ("log_kggrid", "nu_bands", "band_edges")
    original = {key: np.asarray(getattr(ckd.ckd_info, key)) for key in fields}
    parts, records = {}, []
    for end in VARIANTS[variant_name]:
        chunks = completed_chunks(plan, end)
        if any(chunk is None for chunk in chunks):
            raise ValueError(f"Complete all {end} endpoint chunks before merging")
        pieces = {key: [] for key in fields}
        for index, record in enumerate(chunks):
            with np.load(record["files"]["ckd.npz"]["path"], allow_pickle=False) as archive:
                for key in ("T_grid", "P_grid", "ggrid", "weights"):
                    if not np.array_equal(archive[key], getattr(ckd.ckd_info, key)):
                        raise ValueError(f"Endpoint {key} differs from the original table")
                if not np.array_equal(archive["band_edges"], endpoint_grid(plan, end, index)[3]):
                    raise ValueError("Endpoint band edges differ from the pinned plan")
                for key in fields:
                    pieces[key].append(np.asarray(archive[key]))
            records.append(base._file_entry(_directory(plan, end, index) / "complete.json", "Endpoint chunk provenance"))
        parts[end] = {key: np.concatenate(values, axis=-1 if key == "log_kggrid" else 0)
                      for key, values in pieces.items()}
    sequence = ([parts["lower"]] if "lower" in parts else []) + [original] + ([parts["upper"]] if "upper" in parts else [])
    merged = {key: np.concatenate([part[key] for part in sequence], axis=-1 if key == "log_kggrid" else 0)
              for key in fields}
    if not np.array_equal(merged["band_edges"][1:, 0], merged["band_edges"][:-1, 1]):
        raise ValueError("Endpoint bands must meet the original bands exactly")
    ckd.ckd_info = replace(ckd.ckd_info, **merged)
    ckd.nu_bands, ckd.band_edges = merged["nu_bands"], merged["band_edges"]
    ckd._expected_base_meta = dict(
        endpoint_fingerprint=plan["fingerprint"], variant=plan["variants"][variant_name],
        original_ckd=source["files"]["ckd.npz"], endpoint_records=records,
        band_edges_sha256=refinement._array_hash(merged["band_edges"]), limitations=plan["limitations"],
    )
    directory = Path(plan["output_dir"]) / "variants" / variant_name / plan["species"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ckd.npz"
    refinement._save_ckd(ckd, path)
    record = dict(plan=plan, variant=plan["variants"][variant_name], variant_name=variant_name,
                  files={"ckd.npz": base._file_entry(path, "Original CKD plus independently prepared endpoints")},
                  endpoint_records=records, original_ckd=source["files"]["ckd.npz"],
                  original_preparation=base._file_entry(Path(plan["source_preparation_path"]), "Original unchanged LBL preparation"),
                  original_resolved_files={key: source["files"][key] for key in ("diffgrid.npz", "diffgrid_metadata.json")})
    base._write_json(directory / "complete.json", record)
    return directory / "complete.json"


def assemble_manifest(manifest_path, output_root, variant_name, output_path):
    """Require a complete common endpoint variant and preserve every LBL pin."""
    from _rce_wasp18b_assets import load_manifest, load_stellar_spectrum

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new manifest output path")
    manifest = load_manifest(manifest_path)
    config = deepcopy(manifest.config)
    config["files"] = {key: dict(value, path=str(manifest.paths[key])) for key, value in config["files"].items()}
    directory = Path(output_root).resolve() / "variants" / variant_name
    completed = {path.parent.name: path for path in directory.glob("*/complete.json")}
    if set(completed) != set(config["opacity"]["ckd"]) or set(completed) != set(config["opacity"]["lbl"]) or not completed:
        raise ValueError("Endpoint species must exactly match all baseline CKD/LBL species")
    reference, variant = None, None
    for species, path in sorted(completed.items()):
        record = json.loads(path.read_text())
        plan = record["plan"]
        refinement._check_plan(plan["source_plan"])
        if (plan["species"] != species or base._digest_json(plan["identity"]) != plan["fingerprint"]
                or plan["identity"]["source_plan"] != plan["source_plan"]
                or plan["identity"]["driver_sha256"] != base._sha256(Path(__file__))
                or plan["identity"]["refinement_driver_sha256"] != base._sha256(Path(refinement.__file__))
                or record["variant_name"] != variant_name or record["variant"] != plan["variants"][variant_name]):
            raise ValueError("Endpoint provenance does not match the declared variant")
        base._check_files(record["files"])
        base._check_files(dict(record["original_resolved_files"], original=record["original_preparation"], ckd=record["original_ckd"]))
        for item in record["endpoint_records"]:
            base._check_files({"endpoint": item})
        expected_records = []
        for end in VARIANTS[variant_name]:
            chunks = completed_chunks(plan, end)
            if any(chunk is None for chunk in chunks):
                raise ValueError("Missing endpoint chunks")
            expected_records += [base._file_entry(_directory(plan, end, i) / "complete.json", "Endpoint chunk provenance")
                                 for i in range(len(chunks))]
        if record["endpoint_records"] != expected_records:
            raise ValueError("Merged endpoint table references different chunks")
        lbl, ckd = config["opacity"]["lbl"][species], config["opacity"]["ckd"][species]
        matches = {"diffgrid.npz": lbl["asset"], "diffgrid_metadata.json": lbl["metadata_asset"],
                   "original": lbl.get("preparation_asset", f"{species}_preparation"), "ckd": ckd["asset"]}
        entries = dict(record["original_resolved_files"], original=record["original_preparation"], ckd=record["original_ckd"])
        source, _ = _source_records(plan)
        if (source["files"]["ckd.npz"] != record["original_ckd"] or
                any(source["files"][key] != value for key, value in record["original_resolved_files"].items())):
            raise ValueError("Endpoint source assets differ from their original preparation")
        for name, key in matches.items():
            if entries[name]["sha256"] != manifest.sha256[key] or Path(entries[name]["path"]).resolve() != manifest.paths[key]:
                raise ValueError("Endpoint extension uses different original opacity inputs")
        with np.load(record["files"]["ckd.npz"]["path"], allow_pickle=False) as archive:
            grids = {key: np.asarray(archive[key]) for key in ("T_grid", "P_grid", "ggrid", "weights", "band_edges")}
            metadata = json.loads(archive["meta"].tobytes())["base_fingerprint"]
            if metadata["endpoint_records"] != expected_records or metadata["variant"] != record["variant"]:
                raise ValueError("Merged CKD metadata does not match its endpoint records")
        if reference is None:
            reference, variant = grids, record["variant"]
        elif variant != record["variant"] or any(not np.array_equal(grids[key], reference[key]) for key in grids):
            raise ValueError("All endpoint species must use identical grids and variants")
        config["files"][ckd["asset"]] = record["files"]["ckd.npz"]
        config["files"][f"{species}_endpoint_preparation"] = base._file_entry(path, "Endpoint extension provenance")
    load_stellar_spectrum(manifest).validate_coverage(reference["band_edges"][[0, -1], [0, 1]])
    config["spectral_endpoint_extension"] = dict(variant=variant, variant_name=variant_name, limitations=LIMITATIONS,
        baseline_manifest=base._file_entry(manifest.path, "Preserved original interior bands and LBL"))
    config["readiness"] = "Endpoint opacity tables prepared; RCE endpoint sensitivity and UV omission remain unaudited."
    output.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(output, config)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-preparation", type=Path)
    parser.add_argument("--source-preparation", type=Path)
    parser.add_argument("--species", choices=refinement.SPECIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end", choices=("lower", "upper"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--assemble-manifest", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--manifest-output", type=Path)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.assemble_manifest:
            if args.variant is None or args.manifest_output is None or args.dry_run:
                raise ValueError("Assembly requires --variant and --manifest-output, without --dry-run")
            print(assemble_manifest(args.assemble_manifest, args.output_dir, args.variant, args.manifest_output))
            return 0
        if args.base_preparation is None or args.species is None:
            raise ValueError("Preparation requires --base-preparation and --species")
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if not plan["source_merge_ready"]:
            raise ValueError("Complete the original species before endpoint preparation")
        if args.chunk_index is not None:
            if args.end is None:
                raise ValueError("A child chunk requires --end")
            prepare_chunk(plan, args.end, args.chunk_index)
            return 0
        ends = [args.end] if args.end else ["lower", "upper"]
        for end in ends:
            for index, record in enumerate(completed_chunks(plan, end)):
                if record is None:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments,
                                    "--end", end, "--chunk-index", str(index)], check=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        for variant in (ends if args.end else VARIANTS):
            print(merge_variant(plan, variant))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
