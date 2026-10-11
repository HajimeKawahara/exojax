"""Reuse one pinned native teacher for CKD and spectral-sampling diagnostics.

The frozen refinement driver still writes Ng32 and half-width bands. This
wrapper adds Ng16 tables using every second/fourth native spectral sample.
Original completions are never overwritten. Separate sampling fingerprints
describe the subsampling; the original teacher plan remains unchanged.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

import prepare_rce_wasp18b_ckd_refinement as refinement
import prefill_rce_wasp18b_ckd_refinement as prefill


base = refinement.base
_original_write_json = base._write_json
STRIDES = {"sampling128": 4, "sampling256": 2}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _original_write_json(path, value)


def make_plan(args):
    original = refinement.make_plan(argparse.Namespace(
        base_preparation=args.base_preparation, source_preparation=args.source_preparation,
        output_dir=args.refinement_dir, species=args.species,
    ))
    spec = original["source_plan"]["settings"]
    if spec["samples_per_band"] != 512 or spec["ng"] != 16:
        raise ValueError("Sampling variants require the original 512-interval, Ng16 teacher")
    output = args.output_dir.expanduser().resolve()
    if output in (Path(original["output_dir"]), Path(original["source_plan"]["output_dir"])):
        raise ValueError("Use a separate sampling output directory")
    implementation = {
        "wrapper": base._file_entry(Path(__file__), "Actual sampling wrapper"),
        "refinement": base._file_entry(Path(refinement.__file__), "Unchanged refinement driver"),
        "prefill": base._file_entry(Path(prefill.__file__), "Unchanged saved-node eligibility checks"),
    }
    variants = {}
    for name, stride in STRIDES.items():
        contract = dict(teacher_grid_fingerprint=original["source_plan"]["grid_fingerprint"],
                        native_intervals_per_band=512, intervals_per_band=512 // stride,
                        stride=stride, ng=16, band_split=1)
        variants[name] = dict(contract, fingerprint=base._digest_json(contract))
    identity = dict(refinement_fingerprint=original["fingerprint"],
                    implementation=implementation, variants=variants)
    size = (len(spec["temperature_grid_K"]) * len(spec["pressure_grid_bar"])
            * spec["ng"] * spec["number_of_bands"] * 8)
    return dict(schema_version="exojax.rce-wasp18b-ckd-sampling/1",
        identity=identity, fingerprint=base._digest_json(identity), refinement_plan=original,
        variants=variants, species=original["species"], output_dir=str(output),
        number_of_chunks=original["number_of_chunks"],
        storage_estimate=dict(two_variants_chunks_and_merged_coefficient_bytes=4 * size),
        scope="Subsample the existing native teacher; no finer teacher grid or new resolved opacity.")


def _chunk_path(plan, index):
    return Path(plan["output_dir"]) / "_chunks" / plan["species"] / f"chunk_{index:05d}" / "complete.json"


def _existing_refinement(plan, index):
    native = plan["refinement_plan"]
    path = refinement._chunk_path(native, index)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["fingerprint"] != native["fingerprint"] or record["species"] != plan["species"]:
        raise ValueError("Existing refinement has different settings")
    base._check_files(record["files"])
    base._check_files({"source": record["source_chunk"]})
    base._check_sources(record["provenance"]["sources"])
    return record


def completed_chunk(plan, index):
    path = _chunk_path(plan, index)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record["fingerprint"] != plan["fingerprint"] or record["species"] != plan["species"]
            or record["chunk_index"] != index):
        raise ValueError("Existing sampling has different settings or implementation")
    if set(record["files"]) != set(plan["variants"]):
        raise ValueError("Incomplete sampling completion")
    base._check_files(record["files"])
    base._check_files(record["inputs"])
    refinement._source_chunk(plan["refinement_plan"], index)
    return record


class StridedTeacher:
    """Select original grid points without interpolation or a new teacher call."""

    def __init__(self, cached_teacher, stride):
        self.teacher, self.stride = cached_teacher, stride
        self.nu_grid = np.asarray(cached_teacher.nu_grid)[::stride].copy()
        self.method, self.ready = cached_teacher.method, cached_teacher.ready
        self.molmass = getattr(cached_teacher, "molmass", None)

    def xsmatrix(self, temperature, pressure):
        return self.teacher.xsmatrix(temperature, pressure)[:, ::self.stride]

    def meta(self):
        return dict(teacher=self.teacher.meta(), spectral_stride=self.stride)


def _check_grid(plan, index, teacher, edges, temperature, pressure):
    native = plan["refinement_plan"]["source_plan"]
    spec = native["settings"]
    _, expected, _, expected_edges = base.chunk_grid(native, index)
    if (not np.array_equal(teacher.nu_grid, expected) or not np.array_equal(edges, expected_edges)
            or not np.array_equal(temperature, spec["temperature_grid_K"])
            or not np.array_equal(pressure, spec["pressure_grid_bar"])):
        raise ValueError("Sampling requires the exact original P/T/wavenumber/band grids")
    endpoints = np.r_[edges[:, 0], edges[-1, 1]]
    for stride in STRIDES.values():
        selected = expected[::stride]
        if not np.array_equal(selected[::512 // stride], endpoints):
            raise ValueError("Subsampling must preserve every original band endpoint")


def build_sampling(plan, index, cached, edges, temperature, pressure, provenance, execution):
    """Persist only new sampling tables; the native refinement is independent."""
    import jax.numpy as jnp
    from exojax.opacity import OpaCKD

    existing = completed_chunk(plan, index)
    if existing is not None:
        return existing
    _check_grid(plan, index, cached, edges, temperature, pressure)
    source_path, _ = refinement._source_chunk(plan["refinement_plan"], index)
    inputs = dict(plan["identity"]["implementation"],
                  source_chunk=base._file_entry(source_path, "Original completed teacher chunk"),
                  base_preparation=plan["refinement_plan"]["base_preparation"],
                  **execution.get("inputs", {}))
    files, grids = {}, {}
    for name, variant in plan["variants"].items():
        teacher = StridedTeacher(cached, variant["stride"])
        ckd = OpaCKD(teacher, Ng=variant["ng"], band_width=float(np.median(np.diff(edges, axis=1))),
                     band_spacing="log")
        ckd.band_edges = jnp.asarray(edges)
        ckd.nu_bands = jnp.sqrt(ckd.band_edges[:, 0] * ckd.band_edges[:, 1])
        ckd.precompute_tables(np.sort(temperature), pressure)
        grids[name] = refinement._array_hash(teacher.nu_grid)
        ckd.base_opa = None
        ckd._expected_base_meta = dict(
            teacher=provenance, sampling_variant=variant, sampling_execution=execution,
            sampling_wrapper=inputs["wrapper"], native_nu_sha256=refinement._array_hash(cached.nu_grid),
            sampled_nu_sha256=grids[name], temperature_sha256=refinement._array_hash(np.sort(temperature)),
            pressure_sha256=refinement._array_hash(pressure), band_edges_sha256=refinement._array_hash(edges),
        )
        path = Path(plan["output_dir"]) / name / plan["species"] / f"chunk_{index:05d}" / "ckd.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        refinement._save_ckd(ckd, path)
        files[name] = base._file_entry(path, "CKD from a declared subset of the native spectral samples")
    record = dict(fingerprint=plan["fingerprint"], species=plan["species"], chunk_index=index, files=files,
        inputs=inputs, execution=execution, source_grid_fingerprint=plan["refinement_plan"]["source_plan"]["grid_fingerprint"],
        native_nu_sha256=refinement._array_hash(cached.nu_grid), sampled_nu_sha256=grids)
    _write_json(_chunk_path(plan, index), record)
    return record


def prepare_chunk(plan, index):
    """Run the frozen loader once and protect any existing refinement bytes."""
    import jax
    jax.config.update("jax_enable_x64", True)
    existing = _existing_refinement(plan, index)
    if existing is not None and completed_chunk(plan, index) is not None:
        return completed_chunk(plan, index)
    native = plan["refinement_plan"]
    path = refinement._chunk_path(native, index)
    previous = path.read_bytes() if existing is not None else None
    original_build, original_write = refinement.build_variants, base._write_json
    saved = {}

    def build(native_plan, chunk, teacher, edges, temperature, pressure, provenance):
        cached = refinement.CachedTeacher(teacher)
        _check_grid(plan, chunk, cached, edges, temperature, pressure)
        files = (existing["files"] if existing is not None else
                 original_build(native_plan, chunk, cached, edges, temperature, pressure, provenance))
        saved["record"] = build_sampling(plan, chunk, cached, edges, temperature, pressure, provenance,
            dict(method="native_teacher_with_spectral_subsets", native_teacher_recomputed=True,
                 native_teacher_evaluations=1, wrapper_path=str(Path(__file__).resolve())))
        return files

    def write(target, value):
        if previous is not None and Path(target).resolve() == path.resolve():
            if path.read_bytes() != previous:
                raise ValueError("Existing refinement completion changed during sampling")
            return
        original_write(target, value)

    refinement.build_variants, base._write_json = build, write
    try:
        refinement.prepare_chunk(native, index)
    finally:
        refinement.build_variants, base._write_json = original_build, original_write
    return saved["record"]


def prefill_chunk(plan, index):
    """Use exact saved values only after the frozen original-Ng16 check passes."""
    from exojax.opacity import OpaDiffgrid

    if completed_chunk(plan, index) is not None:
        return dict(chunk_index=index, status="skipped_existing")
    started = perf_counter()
    source_path, source = refinement._source_chunk(plan["refinement_plan"], index)
    required = ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json")
    if not all(name in source["files"] for name in required):
        return dict(chunk_index=index, status="ineligible", reason="No original resolved table")
    inputs = {name: source["files"][name] for name in required}
    base._check_files(inputs)
    native = plan["refinement_plan"]["source_plan"]
    _, nu, _, edges = base.chunk_grid(native, index)
    spec = native["settings"]
    temperature, pressure = np.asarray(spec["temperature_grid_K"]), np.asarray(spec["pressure_grid_bar"])
    opacity = OpaDiffgrid.from_saved_opa(inputs["diffgrid.npz"]["path"])
    try:
        teacher = prefill.SavedNodeTeacher(opacity, nu, temperature, pressure, source["provenance"])
        cached = refinement.CachedTeacher(teacher)
        cached.log_floor = teacher.log_floor
        validation = prefill.validate_original_ckd(cached, edges, temperature, pressure,
                                                  inputs["ckd.npz"]["path"], spec["ng"])
    except prefill.IneligibleChunk as error:
        return dict(chunk_index=index, status="ineligible", reason=str(error))
    build_sampling(plan, index, cached, edges, temperature, pressure, source["provenance"],
        dict(method="saved_diffgrid_nodes", native_teacher_recomputed=False,
             wrapper_path=str(Path(__file__).resolve()), inputs=inputs,
             original_ckd_reconstruction=validation))
    return dict(chunk_index=index, status="prefilled", seconds=perf_counter() - started)


def merge_species(plan):
    """Merge sampling coefficients only, retaining exact original LBL pins."""
    from exojax.opacity import OpaCKD

    records = [completed_chunk(plan, i) for i in range(plan["number_of_chunks"])]
    if any(record is None for record in records):
        raise ValueError("Complete every sampling chunk before merging")
    native = plan["refinement_plan"]
    source_path = Path(native["source_preparation_path"])
    source = json.loads(source_path.read_text())
    if source["plan"] != native["source_plan"]:
        raise ValueError("Original merged species does not match the sampling source plan")
    if len(source["chunk_records"]) != len(records):
        raise ValueError("Original and sampling chunk inventories differ")
    for original, record in zip(source["chunk_records"], records):
        actual = record["inputs"]["source_chunk"]
        if (original["sha256"] != actual["sha256"]
                or Path(original["path"]).resolve() != Path(actual["path"]).resolve()):
            raise ValueError("Original merged LBL was built from different teacher chunks")
    resolved = {name: source["files"][name] for name in ("diffgrid.npz", "diffgrid_metadata.json")}
    base._check_files(resolved)
    outputs = []
    for name, variant in plan["variants"].items():
        path = Path(plan["output_dir"]) / name / plan["species"] / "complete.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing["plan"] != plan or existing["variant"] != variant:
                raise ValueError("Existing sampling merge has different provenance")
            base._check_files(existing["files"])
            base._check_files({str(i): item for i, item in enumerate(existing["chunk_records"])})
            outputs.append(str(path))
            continue
        ckd = OpaCKD.load_only().load_tables(records[0]["files"][name]["path"])
        fields = {key: [] for key in ("log_kggrid", "nu_bands", "band_edges")}
        for index, record in enumerate(records):
            with np.load(record["files"][name]["path"], allow_pickle=False) as archive:
                for key in ("T_grid", "P_grid", "ggrid", "weights"):
                    if not np.array_equal(archive[key], getattr(ckd.ckd_info, key)):
                        raise ValueError(f"Sampling chunk {key} differs")
                expected = base.chunk_grid(native["source_plan"], index)[3]
                metadata = json.loads(archive["meta"].tobytes())["base_fingerprint"]
                if (not np.array_equal(archive["band_edges"], expected)
                        or metadata["sampling_variant"] != variant):
                    raise ValueError("Sampling chunk band/variant provenance differs")
                for key in fields:
                    fields[key].append(np.asarray(archive[key]))
        fields = {key: np.concatenate(values, axis=-1 if key == "log_kggrid" else 0)
                  for key, values in fields.items()}
        ckd.ckd_info = replace(ckd.ckd_info, **fields)
        ckd.nu_bands, ckd.band_edges = fields["nu_bands"], fields["band_edges"]
        ckd._expected_base_meta = dict(sampling_variant=variant, sampling_plan_fingerprint=plan["fingerprint"],
            original_preparation=base._file_entry(source_path, "Original matched CKD/LBL preparation"),
            chunks=[record["files"][name]["sha256"] for record in records])
        table_path = path.parent / "ckd.npz"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        refinement._save_ckd(ckd, table_path)
        _write_json(path, dict(plan=plan, variant=variant,
            files={"ckd.npz": base._file_entry(table_path, "Merged CKD spectral-sampling variant")},
            original_preparation=base._file_entry(source_path, "Original unchanged CKD/LBL preparation"),
            original_resolved_files=resolved,
            chunk_records=[base._file_entry(_chunk_path(plan, i), "Sampling chunk provenance") for i in range(len(records))]))
        outputs.append(str(path))
    return outputs


def assemble_manifest(manifest_path, output_root, name, output_path):
    """Replace every CKD species together, preserving all physical/LBL inputs."""
    from _rce_wasp18b_assets import load_manifest

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new manifest path")
    manifest = load_manifest(manifest_path)
    config = deepcopy(manifest.config)
    species = set(config["opacity"]["ckd"])
    directory = Path(output_root).expanduser().resolve() / name
    completed = {path.parent.name: path for path in directory.glob("*/complete.json")}
    if not species or species != set(config["opacity"]["lbl"]) or species != set(completed):
        raise ValueError("Sampling species must exactly match the baseline CKD/LBL mapping")
    config["files"] = {key: dict(value, path=str(manifest.paths[key])) for key, value in config["files"].items()}
    common = None
    for species_name in sorted(species):
        record_path = completed[species_name]
        record = json.loads(record_path.read_text())
        plan, variant = record["plan"], record["variant"]
        native_plan = plan["refinement_plan"]
        native = native_plan["source_plan"]
        refinement._check_plan(native)
        if (plan["species"] != species_name or base._digest_json(plan["identity"]) != plan["fingerprint"]
                or base._digest_json(native_plan["identity"]) != native_plan["fingerprint"]
                or plan["identity"]["refinement_fingerprint"] != native_plan["fingerprint"]
                or native_plan["identity"]["source_plan"] != native
                or variant != plan["variants"][name]
                or variant["teacher_grid_fingerprint"] != config["preparation"]["grid_fingerprint"]):
            raise ValueError("Sampling grid/species provenance differs from the baseline")
        base._check_files(plan["identity"]["implementation"])
        base._check_files(record["files"])
        base._check_files({"original": record["original_preparation"]})
        if len(record["chunk_records"]) != plan["number_of_chunks"]:
            raise ValueError("Sampling merged chunk inventory is incomplete")
        for index, child in enumerate(record["chunk_records"]):
            base._check_files({"child": child})
            chunk = json.loads(Path(child["path"]).read_text())
            if (chunk["fingerprint"] != plan["fingerprint"] or chunk["species"] != species_name
                    or chunk["chunk_index"] != index):
                raise ValueError("Sampling merged chunk identity differs")
            base._check_files(chunk["files"])
            base._check_files(chunk["inputs"])
        lbl = config["opacity"]["lbl"][species_name]
        for filename, key in (("diffgrid.npz", lbl["asset"]), ("diffgrid_metadata.json", lbl["metadata_asset"])):
            entry = record["original_resolved_files"][filename]
            if Path(entry["path"]).resolve() != manifest.paths[key] or entry["sha256"] != manifest.sha256[key]:
                raise ValueError("Sampling and baseline resolved assets differ")
        with np.load(record["files"]["ckd.npz"]["path"], allow_pickle=False) as archive:
            metadata = json.loads(archive["meta"].tobytes())["base_fingerprint"]
            if (metadata["sampling_variant"] != variant or metadata["sampling_plan_fingerprint"] != plan["fingerprint"]
                    or metadata["chunks"] != [json.loads(Path(child["path"]).read_text())["files"][name]["sha256"]
                                               for child in record["chunk_records"]]):
                raise ValueError("Merged sampling metadata differs from its chunk provenance")
            grids = {key: np.asarray(archive[key]) for key in ("T_grid", "P_grid", "ggrid", "weights", "band_edges")}
        old_key = config["opacity"]["ckd"][species_name]["asset"]
        with np.load(manifest.paths[old_key], allow_pickle=False) as archive:
            if any(not np.array_equal(value, archive[key]) for key, value in grids.items()):
                raise ValueError("Sampling must retain all baseline T/P/g/band coordinates")
        if common is None:
            common = variant
        elif common != variant:
            raise ValueError("Sampling species variants differ")
        config["files"][old_key] = record["files"]["ckd.npz"]
        config["opacity"]["ckd"][species_name] = dict(asset=old_key, format="npz")
        config["files"][species_name + "_ckd_sampling"] = base._file_entry(record_path, "Sampling and unchanged resolved provenance")
    config["ckd_sampling"] = dict(variant=common, variant_name=name,
        baseline_manifest=base._file_entry(manifest.path, "Unchanged baseline physical inputs and LBL assets"),
        assembly_driver_sha256=base._sha256(Path(__file__)))
    config["readiness"] = "Sampling subset assembled; native-grid accuracy and P7 acceptance are not certified."
    _write_json(output, config)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-preparation", type=Path)
    parser.add_argument("--source-preparation", type=Path)
    parser.add_argument("--refinement-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--species", choices=refinement.SPECIES)
    parser.add_argument("--chunk-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--prefill-range", type=int, nargs=2, metavar=("START", "STOP"),
                        help="Saved-node chunks START through STOP-1; requires --prefill-only")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assemble-manifest", type=Path)
    parser.add_argument("--variant", choices=sorted(STRIDES))
    parser.add_argument("--manifest-output", type=Path)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.assemble_manifest is not None:
            if args.variant is None or args.manifest_output is None or args.dry_run or args.prefill_only:
                raise ValueError("Assembly requires --variant and --manifest-output")
            print(assemble_manifest(args.assemble_manifest, args.output_dir, args.variant, args.manifest_output))
            return 0
        if args.base_preparation is None or args.species is None or args.refinement_dir is None:
            raise ValueError("Preparation requires base-preparation, species and refinement-dir")
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.chunk_index is not None and not 0 <= args.chunk_index < plan["number_of_chunks"]:
            raise ValueError("Chunk index outside the original plan")
        if args.prefill_range is not None and (not args.prefill_only or args.chunk_index is not None
                or not 0 <= args.prefill_range[0] < args.prefill_range[1] <= plan["number_of_chunks"]):
            raise ValueError("Use a valid prefill range without --chunk-index")
        if args.prefill_only:
            if args.report is None:
                raise ValueError("Saved-node prefill requires a separate --report")
            protected = {args.base_preparation.resolve(), Path(__file__).resolve()}
            if args.source_preparation is not None:
                protected.add(args.source_preparation.resolve())
            if args.report.resolve() in protected:
                raise ValueError("The report must not replace an input")
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            import jax
            jax.config.update("jax_enable_x64", True)
            indices = (range(*args.prefill_range) if args.prefill_range is not None else
                       range(plan["number_of_chunks"]) if args.chunk_index is None else [args.chunk_index])
            report = dict(fingerprint=plan["fingerprint"], chunks=[], status="running")
            for index in indices:
                result = prefill_chunk(plan, index)
                report["chunks"].append(result)
                print(json.dumps(result), flush=True)
                _write_json(args.report, report)
            report["status"] = "complete_prefill_only"
            _write_json(args.report, report)
            return 0
        if args.chunk_index is not None:
            prepare_chunk(plan, args.chunk_index)
            return 0
        if not plan["refinement_plan"]["source_merge_ready"]:
            raise ValueError("Complete the original species before launching all sampling chunks")
        # Finish the ongoing native refinements first; old completed chunks
        # without saved-node sampling are a separate second-pass backlog.
        missing_refinement = [i for i in range(plan["number_of_chunks"]) if _existing_refinement(plan, i) is None]
        missing_sampling = [i for i in range(plan["number_of_chunks"])
                            if i not in missing_refinement and completed_chunk(plan, i) is None]
        for index in missing_refinement + missing_sampling:
            print(f"Sampling {plan['species']} chunk {index + 1}/{plan['number_of_chunks']}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments,
                            "--chunk-index", str(index)], check=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import jax
        jax.config.update("jax_enable_x64", True)
        native = plan["refinement_plan"]
        merged = [Path(native["output_dir"]) / name / plan["species"] / "complete.json" for name in native["variants"]]
        if not all(path.exists() for path in merged):
            if any(path.exists() for path in merged):
                raise ValueError("Partial existing refinement merge requires explicit recovery")
            refinement.merge_species(native, refinement.completed_chunks(native))
        else:
            for path in merged:
                record = json.loads(path.read_text())
                if record["plan"] != native:
                    raise ValueError("Existing refinement merge provenance differs")
                base._check_files(record["files"])
        print(json.dumps(merge_species(plan)), flush=True)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
