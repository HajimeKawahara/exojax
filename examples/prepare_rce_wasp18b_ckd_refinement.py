"""Build two CKD-only refinements from the unchanged, pinned P7 teachers.

Each original chunk is evaluated once for Ng32 and for two half-width bands
at the original Ng. No resolved table is rebuilt. Completed source chunks
must already exist, and merged refinements retain the original LBL references.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_rce_wasp18b_opacity as base
import prepare_rce_wasp18b_supplemental as supplemental
import prepare_rce_wasp18b_atomic as atomic


SPECIES = sorted(set(supplemental.LINE_LISTS) | set(atomic.ATOMIC_LINES))
VARIANTS = {"ng32": dict(ng=32, band_split=1), "bands2": dict(ng=None, band_split=2)}


def _array_hash(array):
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _driver(species):
    if species in atomic.ATOMIC_LINES:
        return atomic, "atomic_jit"
    if species in supplemental.SUPPLEMENTAL_LINE_LISTS:
        return supplemental, "supplemental"
    return base, ""


def _check_plan(plan):
    if base._digest_json(plan["settings"]) != plan["grid_fingerprint"]:
        raise ValueError("Source plan fingerprint does not match its settings")
    if plan["settings"]["script_sha256"] != base._sha256(Path(base.__file__)):
        raise ValueError("The frozen primary preparation driver changed")
    driver, _ = _driver(plan["species"])
    key = "atomic_preparation" if driver is atomic else "supplemental_preparation"
    if driver is not base and plan[key]["driver_sha256"] != base._sha256(Path(driver.__file__)):
        raise ValueError("The frozen species preparation driver changed")


def make_plan(args):
    """Pin the common plan; missing source merges are allowed only for planning."""
    source = args.base_preparation.expanduser().resolve()
    common = json.loads(source.read_text())["plan"]
    _check_plan(common)
    driver, subdirectory = _driver(args.species)
    source_path = (args.source_preparation or Path(common["output_dir"]) / subdirectory
                   / args.species / "complete.json").expanduser().resolve()
    if source_path.exists():
        original = json.loads(source_path.read_text())["plan"]
    else:
        # This also permits a one-chunk smoke while the original full species
        # merge is pending. The completed source chunk is mandatory at execution.
        defaults = driver.parser_with_defaults().parse_args([])
        for name, value in common["settings"].items():
            if hasattr(defaults, name):
                setattr(defaults, name, value)
        defaults.database_root = Path(common["database_root"])
        defaults.inventory = Path(common["inventory"])
        defaults.output_dir = source_path.parent.parent
        defaults.species = args.species
        original = driver.make_plan(defaults)
    _check_plan(original)
    if original["species"] != args.species or original["settings"] != common["settings"]:
        raise ValueError("Selected source species must use the exact common teacher plan")
    spec = common["settings"]
    if spec["ng"] >= 32 or spec["samples_per_band"] % 2:
        raise ValueError("Require original Ng < 32 and an even number of teacher intervals per band")
    output = args.output_dir.expanduser().resolve()
    if output == Path(original["output_dir"]).resolve():
        raise ValueError("Refinement output must differ from the original preparation directory")
    identity = dict(
        teacher_grid_fingerprint=common["grid_fingerprint"],
        teacher_settings=spec, driver_sha256=base._sha256(Path(__file__)),
        base_preparation_sha256=base._sha256(source), source_plan=original,
        variants={name: dict(value, ng=spec["ng"] if value["ng"] is None else value["ng"])
                  for name, value in VARIANTS.items()},
    )
    variants = {name: dict(value, grid_fingerprint=base._digest_json(
        dict(teacher_grid_fingerprint=common["grid_fingerprint"], **value)))
                for name, value in identity["variants"].items()}
    coefficient_bytes = {name: (len(spec["temperature_grid_K"]) * len(spec["pressure_grid_bar"])
                               * value["ng"] * spec["number_of_bands"] * value["band_split"] * 8)
                         for name, value in variants.items()}
    return dict(
        schema_version="exojax.rce-wasp18b-ckd-refinement/1",
        fingerprint=base._digest_json(identity), identity=identity, variants=variants,
        base_preparation=base._file_entry(source, "Pinned original common preparation plan"),
        source_preparation_path=str(source_path), source_merge_ready=source_path.exists(),
        source_plan=original, species=args.species, output_dir=str(output),
        number_of_chunks=original["number_of_chunks"],
        teacher_evaluations_per_chunk=1,
        original_teacher_intervals_per_band=spec["samples_per_band"],
        split_teacher_intervals_per_band=spec["samples_per_band"] // 2,
        storage_estimate=dict(
            merged_coefficient_bytes=coefficient_bytes,
            chunks_and_merged_coefficient_bytes=2 * sum(coefficient_bytes.values()),
            repeat_merge_peak_coefficient_bytes=2 * sum(coefficient_bytes.values()) + max(coefficient_bytes.values()),
            note="Uncompressed coefficients; excludes small grids/metadata and filesystem overhead. Compression usually reduces storage.",
        ),
        note="Shared band endpoints retain the existing CKD builder convention; no new teacher samples or LBL arrays.",
    )


class CachedTeacher:
    """Reuse exactly one host-selected T/P matrix for the paired CKD builders."""

    def __init__(self, teacher):
        self.teacher = teacher
        self.nu_grid = np.asarray(teacher.nu_grid).copy()
        self.method, self.ready = teacher.method, teacher.ready
        self.molmass = getattr(teacher, "molmass", None)
        self._inputs, self._values = None, None

    def meta(self):
        return self.teacher.meta()

    def xsmatrix(self, temperature, pressure):
        if not np.array_equal(self.nu_grid, self.teacher.nu_grid):
            raise ValueError("Cached teacher wavenumbers changed")
        inputs = (np.asarray(temperature), np.asarray(pressure))
        if self._inputs is None:
            self._inputs = tuple(value.copy() for value in inputs)
            self._values = self.teacher.xsmatrix(temperature, pressure)
        elif any(not np.array_equal(actual, expected)
                 for actual, expected in zip(inputs, self._inputs)):
            raise ValueError("Cached teacher requires exactly the same T/P inputs")
        return self._values


def refined_edges(plan, index, core_nu, edges):
    """Split on existing midpoint samples, retaining every original band edge."""
    settings = plan["source_plan"]["settings"]
    samples = settings["samples_per_band"]
    expected = base.chunk_grid(plan["source_plan"], index)
    if not np.array_equal(core_nu, expected[1]) or not np.array_equal(edges, expected[3]):
        raise ValueError("Refinement teacher does not match the pinned original chunk")
    midpoint = core_nu[np.arange(len(edges)) * samples + samples // 2]
    split = np.empty((2 * len(edges), 2))
    split[::2] = np.column_stack((edges[:, 0], midpoint))
    split[1::2] = np.column_stack((midpoint, edges[:, 1]))
    return {"ng32": np.asarray(edges), "bands2": split}


def _save_ckd(ckd, path):
    temporary = path.with_name(path.stem + ".tmp.npz")
    ckd.save_tables(str(temporary), overwrite=True)
    temporary.replace(path)


def build_variants(plan, index, teacher, edges, temperatures, pressure, provenance):
    """Use the ordinary OpaCKD builder twice with one identical teacher matrix."""
    import jax.numpy as jnp
    from exojax.opacity import OpaCKD

    settings = plan["source_plan"]["settings"]
    if (not np.array_equal(temperatures, settings["temperature_grid_K"])
            or not np.array_equal(pressure, settings["pressure_grid_bar"])):
        raise ValueError("Refinement requires the original temperature and pressure grid")
    cache = CachedTeacher(teacher)
    grids = refined_edges(plan, index, cache.nu_grid, edges)
    files = {}
    for name, variant in plan["variants"].items():
        directory = Path(plan["output_dir"]) / name / plan["species"] / f"chunk_{index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        grid = grids[name]
        ckd = OpaCKD(cache, Ng=variant["ng"], band_width=float(np.median(np.diff(grid, axis=1))),
                     band_spacing="log")
        ckd.band_edges = jnp.asarray(grid)
        ckd.nu_bands = jnp.sqrt(ckd.band_edges[:, 0] * ckd.band_edges[:, 1])
        ckd.precompute_tables(np.sort(temperatures), pressure)
        # Serialize explicit teacher provenance and the changed CKD grid, not
        # the cache's mutable runtime state or a false original grid identity.
        ckd.base_opa = None
        ckd._expected_base_meta = dict(
            teacher=provenance, refinement=variant,
            refinement_driver_sha256=plan["identity"]["driver_sha256"],
            teacher_nu_sha256=_array_hash(cache.nu_grid),
            temperature_sha256=_array_hash(np.sort(temperatures)),
            pressure_sha256=_array_hash(pressure), band_edges_sha256=_array_hash(grid),
        )
        path = directory / "ckd.npz"
        _save_ckd(ckd, path)
        files[name] = base._file_entry(path, "CKD-only refinement from the pinned original teacher")
    return files


def _chunk_path(plan, index):
    return Path(plan["output_dir"]) / "_chunks" / plan["species"] / f"chunk_{index:05d}" / "complete.json"


def _source_chunk(plan, index):
    path = Path(plan["source_preparation_path"]).parent / f"chunk_{index:05d}" / "complete.json"
    record = json.loads(path.read_text())
    original = plan["source_plan"]
    if (record["grid_fingerprint"] != original["grid_fingerprint"]
            or record["mdb_path"] != original["mdb_path"]):
        raise ValueError("Original chunk grid or line list differs from the source plan")
    for key in ("atomic_preparation", "supplemental_preparation"):
        if record["provenance"].get(key) != original.get(key):
            raise ValueError("Original chunk physics/driver differs from the source plan")
    base._check_sources(record["provenance"]["sources"])
    return path, record


def prepare_chunk(plan, index):
    """Intercept only table persistence while retaining the frozen line loaders."""
    import jax

    jax.config.update("jax_enable_x64", True)
    source_path, source = _source_chunk(plan, index)
    original = deepcopy(plan["source_plan"])
    original["output_dir"] = str(Path(plan["output_dir"]) / "_chunks")
    driver, _ = _driver(plan["species"])
    build_tables, write_json = base.build_tables, base._write_json
    saved = {}

    def paired(teacher, edges, temperatures, pressure, ng, directory, provenance, resolved_range=None):
        # The frozen loaders must identify the same actual input bytes. Their
        # stat timestamps may differ after a verified copy of an unchanged file.
        for key in ("number_of_lines", "no_lines_reason", "profile_kernel", "wing_padding_cm1"):
            if provenance.get(key) != source["provenance"].get(key):
                raise ValueError(f"Original teacher provenance changed: {key}")
        hashes = lambda values: {key: value["sha256"] for key, value in values.items()}
        if hashes(provenance["sources"]) != hashes(source["provenance"]["sources"]):
            raise ValueError("Original teacher source hashes changed")
        saved["files"] = build_variants(plan, index, teacher, edges, temperatures, pressure, provenance)
        saved["provenance"] = provenance

    def capture_complete(path, value):
        if Path(path).name != "complete.json":
            write_json(path, value)

    base.build_tables, base._write_json = paired, capture_complete
    try:
        driver.prepare_chunk(original, index)
    finally:
        base.build_tables, base._write_json = build_tables, write_json
    record = dict(fingerprint=plan["fingerprint"], species=plan["species"],
                  source_chunk=base._file_entry(source_path, "Original completed teacher chunk"),
                  files=saved["files"], provenance=saved["provenance"])
    base._write_json(_chunk_path(plan, index), record)
    return record


def completed_chunks(plan):
    records = []
    for index in range(plan["number_of_chunks"]):
        path = _chunk_path(plan, index)
        if not path.exists():
            records.append(None)
            continue
        record = json.loads(path.read_text())
        if record["fingerprint"] != plan["fingerprint"] or record["species"] != plan["species"]:
            raise ValueError("Existing refinement has different settings; choose a new output directory")
        base._check_files(record["files"])
        base._check_files({"source_chunk": record["source_chunk"]})
        base._check_sources(record["provenance"]["sources"])
        records.append(record)
    return records


def merge_species(plan, records):
    """Merge CKD only, preserving pinned references to the original LBL files."""
    from exojax.opacity import OpaCKD

    source_path = Path(plan["source_preparation_path"])
    source = json.loads(source_path.read_text())
    if source["plan"] != plan["source_plan"] or any(record is None for record in records):
        raise ValueError("Require all refinement chunks and the matching complete original species")
    if len(source["chunk_records"]) != len(records) or len(records) != plan["number_of_chunks"]:
        raise ValueError("Original and refined chunk inventories differ")
    for original, refined in zip(source["chunk_records"], records):
        if (original["sha256"] != refined["source_chunk"]["sha256"]
                or Path(original["path"]).resolve() != Path(refined["source_chunk"]["path"]).resolve()):
            raise ValueError("Original merged LBL was built from different teacher chunks")
    # Hash original references without loading their large coefficient arrays.
    original_files = {name: source["files"][name] for name in ("diffgrid.npz", "diffgrid_metadata.json")}
    base._check_files(original_files)
    outputs = []
    for name, variant in plan["variants"].items():
        ckd = OpaCKD.load_only().load_tables(records[0]["files"][name]["path"])
        fields = {key: [] for key in ("log_kggrid", "nu_bands", "band_edges")}
        previous = None
        for index, record in enumerate(records):
            with np.load(record["files"][name]["path"], allow_pickle=False) as archive:
                for key in ("T_grid", "P_grid", "ggrid", "weights"):
                    if not np.array_equal(archive[key], getattr(ckd.ckd_info, key)):
                        raise ValueError(f"Inconsistent chunk {key}")
                edges = archive["band_edges"]
                expected = base.chunk_grid(plan["source_plan"], index)
                if not np.array_equal(edges, refined_edges(plan, index, expected[1], expected[3])[name]):
                    raise ValueError("Refinement chunk edges do not match the pinned grid")
                if previous is not None and previous != edges[0, 0]:
                    raise ValueError("Refinement chunks do not share their exact band endpoint")
                previous = edges[-1, 1]
                for key in fields:
                    fields[key].append(np.asarray(archive[key]))
        fields = {key: np.concatenate(values, axis=-1 if key == "log_kggrid" else 0)
                  for key, values in fields.items()}
        ckd.ckd_info = replace(ckd.ckd_info, **fields)
        ckd.nu_bands, ckd.band_edges = fields["nu_bands"], fields["band_edges"]
        ckd._expected_base_meta = dict(
            refinement=variant, teacher_grid_fingerprint=plan["source_plan"]["grid_fingerprint"],
            source_preparation=base._file_entry(source_path, "Original matched CKD and LBL preparation"),
            band_edges_sha256=_array_hash(fields["band_edges"]),
            chunks=[record["files"][name]["sha256"] for record in records],
        )
        directory = Path(plan["output_dir"]) / name / plan["species"]
        path = directory / "ckd.npz"
        _save_ckd(ckd, path)
        complete = dict(plan=plan, variant=variant,
                        files={"ckd.npz": base._file_entry(path, "Merged CKD-only refinement")},
                        original_resolved_files=original_files,
                        original_preparation=base._file_entry(source_path, "Original unchanged resolved opacity preparation"),
                        chunk_records=[base._file_entry(_chunk_path(plan, i), "Paired refinement provenance")
                                       for i in range(len(records))])
        complete_path = directory / "complete.json"
        base._write_json(complete_path, complete)
        outputs.append(str(complete_path))
    return outputs


def assemble_manifest(manifest_path, output_root, variant_name, output_path):
    """Replace every CKD species together, preserving the original LBL pins."""
    from _rce_wasp18b_assets import load_manifest

    output = Path(output_path).expanduser().resolve()
    if output.exists() or output == Path(manifest_path).expanduser().resolve():
        raise ValueError("Choose a new manifest output path; existing files are not overwritten")
    manifest = load_manifest(manifest_path)
    config = deepcopy(manifest.config)
    species = set(config["opacity"]["ckd"])
    directory = Path(output_root).expanduser().resolve() / variant_name
    completed = {path.parent.name: path for path in directory.glob("*/complete.json")}
    if not species or species != set(config["opacity"]["lbl"]) or species != set(completed):
        raise ValueError("Refinement species must exactly match all CKD/LBL species in the manifest")
    config["files"] = {key: dict(value, path=str(manifest.paths[key]))
                       for key, value in config["files"].items()}
    common_variant, common_grids = None, None
    for name in sorted(species):
        record_path = completed[name]
        record = json.loads(record_path.read_text())
        plan, variant = record["plan"], record["variant"]
        _check_plan(plan["source_plan"])
        if (plan["species"] != name or base._digest_json(plan["identity"]) != plan["fingerprint"]
                or plan["identity"]["source_plan"] != plan["source_plan"]
                or variant != plan["variants"][variant_name]
                or plan["source_plan"]["grid_fingerprint"] != config["preparation"]["grid_fingerprint"]):
            raise ValueError("Refinement grid/species provenance does not match the baseline")
        if common_variant is None:
            common_variant = variant
        elif variant != common_variant:
            raise ValueError("All CKD species must use the same refinement variant")
        if set(record["files"]) != {"ckd.npz"}:
            raise ValueError("Refinement records must contain only the new CKD table")
        if plan["base_preparation"]["sha256"] != plan["identity"]["base_preparation_sha256"]:
            raise ValueError("Pinned common preparation identity changed")
        base._check_files({"base_preparation": plan["base_preparation"]})
        base._check_files(record["files"])
        base._check_files({"original_preparation": record["original_preparation"]})
        base._check_files(record["original_resolved_files"])
        if len(record["chunk_records"]) != plan["number_of_chunks"]:
            raise ValueError("Incomplete refinement chunk inventory")
        for index, item in enumerate(record["chunk_records"]):
            if Path(item["path"]).resolve() != _chunk_path(plan, index).resolve():
                raise ValueError("Refinement chunk record paths differ from the plan")
            base._check_files({"chunk": item})
        chunks = completed_chunks(plan)
        if any(chunk is None for chunk in chunks):
            raise ValueError("Missing refinement chunks")
        original = json.loads(Path(record["original_preparation"]["path"]).read_text())
        if (original["plan"] != plan["source_plan"]
                or len(original["chunk_records"]) != len(chunks)
                or any(first["sha256"] != second["source_chunk"]["sha256"]
                       for first, second in zip(original["chunk_records"], chunks))):
            raise ValueError("Original LBL and refined CKD have different teacher provenance")
        lbl = config["opacity"]["lbl"][name]
        original_key = lbl.get("preparation_asset", f"{name}_preparation")
        required = {"diffgrid.npz": lbl["asset"], "diffgrid_metadata.json": lbl["metadata_asset"]}
        entries = dict(record["original_resolved_files"], original=record["original_preparation"])
        for field, key in dict(required, original=original_key).items():
            if (entries[field]["sha256"] != manifest.sha256[key]
                    or Path(entries[field]["path"]).resolve() != manifest.paths[key]):
                raise ValueError("Refinement would reuse a different original LBL preparation")
        with np.load(record["files"]["ckd.npz"]["path"], allow_pickle=False) as archive:
            metadata = json.loads(archive["meta"].tobytes())["base_fingerprint"]
            if (metadata["refinement"] != variant
                    or metadata["chunks"] != [chunk["files"][variant_name]["sha256"] for chunk in chunks]):
                raise ValueError("Merged CKD metadata identifies different refinement chunks")
            grids = {key: np.asarray(archive[key]) for key in ("T_grid", "P_grid", "ggrid", "weights", "band_edges")}
            spec = plan["source_plan"]["settings"]
            if (len(grids["ggrid"]) != variant["ng"]
                    or not np.array_equal(grids["T_grid"], np.sort(spec["temperature_grid_K"]))
                    or not np.array_equal(grids["P_grid"], spec["pressure_grid_bar"])):
                raise ValueError("Refined CKD temperature/pressure/g grid does not match the teacher")
            expected = []
            for index in range(plan["number_of_chunks"]):
                _, nu, _, edges = base.chunk_grid(plan["source_plan"], index)
                expected.append(refined_edges(plan, index, nu, edges)[variant_name])
            if not np.array_equal(grids["band_edges"], np.concatenate(expected)):
                raise ValueError("Refined CKD band edges differ from the declared variant")
        if common_grids is None:
            common_grids = grids
        elif any(not np.array_equal(grids[key], common_grids[key]) for key in grids):
            raise ValueError("Refinement species use different CKD grids")
        ckd_key = config["opacity"]["ckd"][name]["asset"]
        config["opacity"]["ckd"][name] = dict(asset=ckd_key, format="npz")
        config["files"][ckd_key] = record["files"]["ckd.npz"]
        config["files"][f"{name}_ckd_refinement"] = base._file_entry(record_path, "CKD refinement and unchanged LBL provenance")
    config["ckd_refinement"] = dict(
        variant=common_variant, variant_name=variant_name,
        baseline_manifest=base._file_entry(manifest.path, "Unchanged physical inputs and original resolved opacities"),
        assembly_driver_sha256=base._sha256(Path(__file__)),
    )
    config["readiness"] = "CKD refinement assembled with unchanged resolved opacities; RCE convergence comparison remains required."
    output.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(output, config)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-preparation", type=Path)
    parser.add_argument("--source-preparation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--species", choices=SPECIES)
    parser.add_argument("--assemble-manifest", type=Path, help="Assemble this baseline after every CKD species is refined")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.assemble_manifest is not None:
            if args.variant is None or args.manifest_output is None or args.dry_run or args.chunk_index is not None:
                raise ValueError("Assembly requires --variant and --manifest-output; omit --dry-run and --chunk-index")
            print(assemble_manifest(args.assemble_manifest, args.output_dir, args.variant, args.manifest_output))
            return 0
        if args.base_preparation is None or args.species is None:
            raise ValueError("Preparation requires --base-preparation and --species")
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.chunk_index is not None:
            prepare_chunk(plan, args.chunk_index)
            return 0
        if not plan["source_merge_ready"]:
            raise ValueError("Complete the original species before launching all refinement chunks")
        records = completed_chunks(plan)
        for index, record in enumerate(records):
            if record is None:
                print(f"Refining {args.species} chunk {index + 1}/{len(records)}", flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments,
                                "--chunk-index", str(index)], check=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import jax
        jax.config.update("jax_enable_x64", True)
        print(json.dumps(merge_species(plan, completed_chunks(plan)), indent=2))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
