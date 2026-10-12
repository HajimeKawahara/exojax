"""Prefill eligible CKD refinements from unchanged saved Diffgrid nodes.

Only complete original chunk grids with no floor-clipped samples qualify.
Reconstructing the original CKD table must agree within 1e-12 in log(k).
This avoids native teacher recomputation without changing its physical inputs
or samples. Existing outputs are never overwritten. Coordinate with the native
preparation job before prefilling its output tree; restart it afterward so its
initial inventory recognizes newly completed chunks.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import prepare_rce_wasp18b_ckd_refinement as refinement
from exojax.opacity import OpaCKD, OpaDiffgrid


LOG_K_TOLERANCE = 1e-12
base = refinement.base


class IneligibleChunk(ValueError):
    """The original native teacher must be used for this chunk."""


class SavedNodeTeacher:
    """Exact T/P lookup at saved nodes, with no opacity interpolation."""

    method, ready = "saved-diffgrid-nodes", True

    def __init__(self, opacity, core_nu, temperatures, pressure, provenance):
        info = opacity.diffgrid_info
        self.nu_grid = np.asarray(opacity.nu_grid)
        self.temperature = np.asarray(info.temperature_grid)
        self.pressure = np.asarray(info.pressure_grid)
        self.log_values = np.asarray(info.log_cross_section_grid)
        self.provenance = provenance
        if not np.array_equal(self.nu_grid, core_nu):
            raise IneligibleChunk("Saved Diffgrid does not cover the exact full chunk core")
        if (not np.array_equal(self.temperature, np.sort(temperatures)[::-1])
                or not np.array_equal(self.pressure, pressure)):
            raise IneligibleChunk("Saved temperature/pressure nodes differ from the original plan")
        if self.log_values.dtype != np.float64 or not np.all(np.isfinite(self.log_values)):
            raise IneligibleChunk("Saved log cross sections must be finite float64 values")
        self.log_floor = float(info.log_cross_section_floor)
        if not np.all(self.log_values > self.log_floor):
            raise IneligibleChunk("Floor-clipped samples cannot reconstruct the original teacher")

    def meta(self):
        return self.provenance

    def xsmatrix(self, temperature, pressure):
        temperature, pressure = np.asarray(temperature), np.asarray(pressure)
        if temperature.ndim != 1 or temperature.shape != pressure.shape:
            raise ValueError("Use matching one-dimensional T/P node coordinates")
        order = np.argsort(self.temperature)
        ti = np.searchsorted(self.temperature[order], temperature)
        pi = np.searchsorted(self.pressure, pressure)
        if (np.any(ti >= len(order)) or np.any(pi >= len(self.pressure))
                or not np.array_equal(self.temperature[order[ti]], temperature)
                or not np.array_equal(self.pressure[pi], pressure)):
            raise ValueError("Saved-node teacher accepts only exact original T/P nodes")
        return jnp.asarray(np.exp(self.log_values[pi, order[ti]]))


def validate_original_ckd(teacher, edges, temperatures, pressure, original_path, ng):
    """Reconstruct every original log(k) value using the ordinary CKD builder."""
    original = OpaCKD.load_only().load_tables(str(original_path))
    reconstructed = OpaCKD(teacher, Ng=ng, band_width=float(np.median(np.diff(edges, axis=1))))
    reconstructed.band_edges = jnp.asarray(edges)
    reconstructed.nu_bands = jnp.sqrt(reconstructed.band_edges[:, 0] * reconstructed.band_edges[:, 1])
    reconstructed.precompute_tables(np.sort(temperatures), pressure)
    for name in ("T_grid", "P_grid", "ggrid", "weights", "band_edges", "nu_bands"):
        if not np.array_equal(getattr(original.ckd_info, name), getattr(reconstructed.ckd_info, name)):
            raise IneligibleChunk(f"Original CKD {name} differs from reconstructed nodes")
    actual, expected = np.asarray(reconstructed.ckd_info.log_kggrid), np.asarray(original.ckd_info.log_kggrid)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual) & np.isfinite(expected)):
        raise IneligibleChunk("Original and reconstructed CKD arrays must be finite and matching")
    maximum = float(np.max(np.abs(actual - expected)))
    if maximum > LOG_K_TOLERANCE:
        raise IneligibleChunk(f"Original CKD reconstruction exceeds log(k) tolerance: {maximum}")
    return dict(ng=int(ng), max_absolute_log_k_difference=maximum,
                absolute_log_k_tolerance=LOG_K_TOLERANCE, compared_values=int(actual.size),
                all_saved_values_strictly_above_floor=True, log_cross_section_floor=teacher.log_floor)


def prefill_chunk(plan, index, *, dry_run=False):
    """Create a compatible paired completion, or leave native work untouched."""
    started = perf_counter()
    destination = refinement._chunk_path(plan, index)
    directories = [destination.parent] + [Path(plan["output_dir"]) / name / plan["species"]
                   / f"chunk_{index:05d}" for name in plan["variants"]]
    result = dict(chunk_index=index, species=plan["species"])
    if any(path.exists() for path in directories):
        return dict(result, status="skipped_existing", reason="Existing completion or partial output is never overwritten")
    source_path, source = refinement._source_chunk(plan, index)
    required = ("diffgrid.npz", "diffgrid_metadata.json", "ckd.npz")
    if not all(name in source["files"] for name in required):
        return dict(result, status="ineligible", reason="No original resolved table for this chunk")
    inputs = {name: source["files"][name] for name in required}
    base._check_files(inputs)
    _, core_nu, _, edges = base.chunk_grid(plan["source_plan"], index)
    spec = plan["source_plan"]["settings"]
    temperature, pressure = np.asarray(spec["temperature_grid_K"]), np.asarray(spec["pressure_grid_bar"])
    opacity = OpaDiffgrid.from_saved_opa(inputs["diffgrid.npz"]["path"])
    try:
        teacher = SavedNodeTeacher(opacity, core_nu, temperature, pressure, source["provenance"])
        validation = validate_original_ckd(teacher, edges, temperature, pressure,
                                          inputs["ckd.npz"]["path"], spec["ng"])
    except IneligibleChunk as error:
        return dict(result, status="ineligible", reason=str(error), seconds=perf_counter() - started)
    execution = dict(method="saved_diffgrid_nodes", native_teacher_recomputed=False,
        inputs=dict(inputs, prefill_driver=base._file_entry(Path(__file__), "Actual saved-node prefill implementation")),
        original_ckd_reconstruction=validation)
    if dry_run:
        return dict(result, status="eligible", execution=execution, seconds=perf_counter() - started)
    provenance = deepcopy(source["provenance"])
    provenance["refinement_execution"] = execution
    # Existing frozen completion checks also validate these actual execution
    # inputs on resume, using their hashes when file metadata changes.
    for entry in execution["inputs"].values():
        path = Path(entry["path"])
        stat = path.stat()
        provenance["sources"][str(path.resolve())] = dict(entry, stat=[stat.st_size, stat.st_mtime_ns])
    teacher.provenance = provenance
    # Recheck before creating files; callers must avoid concurrent native work
    # on the same chunk, since the frozen native writer has no shared lock.
    if any(path.exists() for path in directories):
        return dict(result, status="skipped_existing", reason="Output appeared during validation")
    destination.parent.mkdir(parents=True, exist_ok=False)
    files = refinement.build_variants(plan, index, teacher, edges, temperature, pressure, provenance)
    record = dict(fingerprint=plan["fingerprint"], species=plan["species"],
        source_chunk=base._file_entry(source_path, "Original completed teacher chunk"),
        files=files, provenance=provenance)
    base._write_json(destination, record)
    return dict(result, status="prefilled", completion=str(destination), execution=execution,
                seconds=perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-preparation", type=Path, required=True)
    parser.add_argument("--source-preparation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--species", choices=refinement.SPECIES, required=True)
    parser.add_argument("--chunk-index", type=int, action="append", help="Repeat to select chunks; default checks all chunks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.resolve() in {path.resolve() for path in
            (args.base_preparation, args.source_preparation, Path(__file__)) if path is not None}:
        parser.error("The report must be separate from its inputs")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    plan = refinement.make_plan(args)
    indices = args.chunk_index if args.chunk_index is not None else range(plan["number_of_chunks"])
    if any(index < 0 or index >= plan["number_of_chunks"] for index in indices):
        parser.error("Chunk index outside the original plan")
    report = dict(fingerprint=plan["fingerprint"], prefill_driver_sha256=base._sha256(Path(__file__)),
                  output_dir=plan["output_dir"], chunks=[], status="running")
    try:
        for index in indices:
            result = prefill_chunk(plan, index, dry_run=args.dry_run)
            report["chunks"].append(result)
            print(json.dumps(result), flush=True)
            base._write_json(args.report, report)
        report["status"] = "complete_prefill_only"
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        base._write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
