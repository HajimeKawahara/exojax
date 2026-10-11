"""Prepare matched Fe I, Na I and K I tables with native atomic line physics.

The unchanged molecular driver defines the common spectral/temperature grid.
Atomic assumptions are recorded separately; its PreMODIT settings do not
describe this teacher. Each chunk uses bounded batches of native line profiles.
Downloads are explicit and reusable. --dry-run performs no writes or downloads.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_rce_wasp18b_opacity as base


ATOMIC_LINES = {"Fe1": (26, "gf2600.all"), "Na1": (11, "gf1100.all"),
                "K1": (19, "gf1900.all")}
SOURCE_ROOT = "http://kurucz.harvard.edu/linelists/gfall/"


def parser_with_defaults():
    parser = base.parser_with_defaults()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest in ("species", "assemble_species"):
            action.choices = sorted(ATOMIC_LINES)
    parser.set_defaults(species="Fe1")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--line-batch-size", type=int, default=128)
    parser.add_argument("--atomic-perturbers", nargs=3, type=float, default=[0., .16, .84],
                        metavar=("H", "He", "H2"), help="Fixed number fractions for native atomic broadening.")
    parser.add_argument("--atomic-line-padding", type=float,
                        help="Line-center halo in cm-1; defaults to 25 for Fe, 9000 for Na/K.")
    parser.add_argument("--line-profile", choices=("auto", "voigt", "alkali_subvoigt"), default="auto")
    return parser


def make_plan(args):
    args = argparse.Namespace(**vars(args))
    root = args.database_root.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = root / "rce_wasp18b/prepared/atomic_jit"
    if args.mdb_path is None:
        args.mdb_path = root / "Kurucz" / ATOMIC_LINES[args.species][1]
    fractions = np.asarray(args.atomic_perturbers, dtype=float)
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1., rtol=0., atol=1e-12):
        raise ValueError("Atomic perturber fractions must be nonnegative and sum to one")
    if args.line_batch_size < 1:
        raise ValueError("Line batch size must be positive")
    profile = args.line_profile
    if profile == "auto":
        profile = "voigt" if args.species == "Fe1" else "alkali_subvoigt"
    if profile == "alkali_subvoigt" and args.species == "Fe1":
        raise ValueError("Sub-Voigt alkali profiles require neutral Na or K")
    padding = args.atomic_line_padding
    if padding is None:
        padding = 9000. if profile == "alkali_subvoigt" else args.wing_padding
    if not np.isfinite(padding) or padding < 0 or (profile == "alkali_subvoigt" and padding < 9000.):
        raise ValueError("Atomic line padding must be nonnegative and at least 9000 cm-1 for sub-Voigt alkalis")
    plan = base.make_plan(args)
    plan["atomic_preparation"] = dict(
        driver_sha256=base._sha256(Path(__file__).resolve()),
        base_script_sha256=base._sha256(Path(base.__file__).resolve()),
        species=args.species, source_url=SOURCE_ROOT + ATOMIC_LINES[args.species][1],
        teacher_method="atomic_lpf_batched", line_profile=profile,
        line_batch_size=args.line_batch_size, line_padding_cm1=padding,
        perturber_fractions_H_He_H2=fractions.tolist(),
        broadening="Native gamma_vald3; fixed H/He/H2 fractions, natural widths, Unsold fallback, zero Stark electron density",
        partition_function="Barklem & Collet (2016); interpolated, not extrapolated",
        completeness="Not established by table temperature extent; ionized species and bound-free absorption excluded",
    )
    return plan


def download_lines(plan):
    """Reuse a nonempty file, or download the published single-species list."""
    path = Path(plan["mdb_path"])
    if not path.is_file():
        from exojax.database.exomol._pyexocross_download import _download_file
        path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(plan["atomic_preparation"]["source_url"], path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty atomic line list: {path}")
    return path


class BatchedAtomicTeacher:
    """Sum native profiles without materializing every line against the grid."""

    method = "atomic_lpf_batched"
    ready = True

    def __init__(self, adb, nu_grid, species, profile, batch_size):
        import jax
        import jax.numpy as jnp

        self.adb, self.nu_grid = adb, np.asarray(nu_grid)
        self.profile, self.species = profile, species
        self.batch_size = min(batch_size, len(adb.nu_lines))
        self.blocks = (len(adb.nu_lines) + self.batch_size - 1) // self.batch_size
        self.padding = self.blocks * self.batch_size - len(adb.nu_lines)
        self.centers = jnp.pad(adb.dev_nu_lines, (0, self.padding), constant_values=nu_grid[0]).reshape(self.blocks, self.batch_size)
        self._compiled_xsmatrix = jax.jit(self._xsmatrix)

    def xsmatrix(self, temperature, pressure):
        # Reuse the line-scan executable across the teacher's layer batches.
        return self._compiled_xsmatrix(temperature, pressure)

    def _xsmatrix(self, temperature, pressure):
        import jax
        import jax.numpy as jnp
        from exojax.opacity.lpf import lpf
        from exojax.opacity.alkali import _xsvector

        adb = self.adb
        strengths, gamma, sigma = lpf.vald(
            adb, temperature, pressure * adb.vmrH,
            pressure * adb.vmrHe, pressure * adb.vmrHH)

        def blocks(values, fill):
            return jnp.pad(values, ((0, 0), (0, self.padding)), constant_values=fill).reshape(
                len(temperature), self.blocks, self.batch_size).transpose(1, 0, 2)

        def add_lines(total, line_data):
            centers, widths, damping, intensities = line_data
            offsets = jnp.asarray(self.nu_grid)[None, :] - centers[:, None]
            if self.profile == "voigt":
                cross_section = lpf.xsmatrix(offsets, widths, damping, intensities)
            else:
                detuning, cutoff = (30., 5000.) if self.species == "Na1" else (20., 1600.)
                cross_section = jax.vmap(_xsvector, (None, 0, 0, 0, 0, None, None))(
                    offsets, widths, damping, intensities, temperature, detuning, cutoff)
            return total + cross_section, None

        result, _ = jax.lax.scan(add_lines,
            jnp.zeros((len(temperature), len(self.nu_grid)), dtype=temperature.dtype),
            (self.centers, blocks(sigma, 1.), blocks(gamma, 0.), blocks(strengths, 0.)))
        return result


def _source_files(database):
    """Pin the raw lines and the installed atomic data/physics implementation."""
    import exojax

    root = Path(exojax.__file__).resolve().parent
    paths = [database] + list((root / "data/atom").glob("*.txt"))
    paths += list((root / "database/core_atom").glob("*.py"))
    paths += [root / name for name in (
        "database/kurucz/api.py", "opacity/lpf/lpf.py", "opacity/alkali.py",
        "database/core/line_strength.py", "database/core/broadening.py")]
    records = {}
    for path in paths:
        stat = path.stat()
        records[str(path)] = dict(path=str(path), sha256=base._sha256(path),
                                  stat=[stat.st_size, stat.st_mtime_ns])
    return records


def prepare_chunk(plan, index):
    import jax
    from exojax.database.core_atom.io import load_atomicdata
    from exojax.database.kurucz.api import AdbKurucz

    jax.config.update("jax_enable_x64", True)
    grid, core_nu, selection, edges = base.chunk_grid(plan, index)
    database = download_lines(plan)
    spec, atomic = plan["settings"], plan["atomic_preparation"]
    padding = atomic["line_padding_cm1"]
    adb = AdbKurucz(database, nurange=[max(0., grid[0] - padding), grid[-1] + padding],
                    gpu_transfer=False, vmr_fraction=atomic["perturber_fractions_H_He_H2"])
    if np.min(adb.T_gQT) > spec["temperature_min"] or np.max(adb.T_gQT) < spec["temperature_max"]:
        raise ValueError("Atomic partition-function table does not cover the requested temperatures")
    if np.any(adb._ielem != ATOMIC_LINES[plan["species"]][0]) or np.any(adb._iion != 1):
        raise ValueError("The atomic file must contain only the requested neutral species")
    if np.any(adb._elower < 0) or np.any(adb._eupper < 0):
        raise ValueError("Signed predicted energies need explicit handling before using this reader")
    teacher, no_lines_reason = None, None
    if len(adb.nu_lines):
        adb.generate_jnp_arrays()
        teacher = BatchedAtomicTeacher(adb, grid, plan["species"], atomic["line_profile"], atomic["line_batch_size"])
    else:
        no_lines_reason = "No line centers within the atomic line halo; zero line opacity assumed."
    provenance = dict(
        grid_fingerprint=plan["grid_fingerprint"], species=plan["species"],
        line_list=str(database), sources=_source_files(database), profile_kernel=None,
        atomic_preparation=atomic, no_lines_reason=no_lines_reason,
        number_of_lines=len(adb.nu_lines), wing_padding_cm1=spec["wing_padding"],
        line_completeness=atomic["completeness"],
    )
    directory = Path(plan["output_dir"]) / plan["species"] / f"chunk_{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    teacher = base.CroppedTeacher(teacher, core_nu, selection, provenance)
    teacher.molmass = float(load_atomicdata().set_index("ielem").loc[ATOMIC_LINES[plan["species"]][0], "mass"])
    base.build_tables(teacher, edges, np.asarray(spec["temperature_grid_K"]),
        np.asarray(spec["pressure_grid_bar"]), spec["ng"], directory, provenance,
        (spec["resolved_nu_min"], spec["resolved_nu_max"]))
    files = {name: base._file_entry(directory / name, "ExoJAX native atomic preparation")
             for name in ("ckd.npz", "diffgrid.npz", "diffgrid_metadata.json") if (directory / name).exists()}
    base._write_json(directory / "complete.json", dict(grid_fingerprint=plan["grid_fingerprint"],
        mdb_path=plan["mdb_path"], files=files, provenance=provenance))


def completed_chunks(plan):
    records = base.completed_chunks(plan)
    for record in records:
        if record is not None and record["provenance"].get("atomic_preparation") != plan["atomic_preparation"]:
            raise ValueError("Existing atomic preparation uses different physics or code; choose a new output directory")
    return records


def merge_species(plan, records):
    """Reuse the host merger, then correct its molecular-only metadata labels."""
    base.merge_species(plan, records)
    directory = Path(plan["output_dir"]) / plan["species"]
    metadata_path = directory / "diffgrid_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["opa_state"].update(teacher_method="atomic_lpf_batched", teacher_profile_kernel=None)
    base._write_json(metadata_path, metadata)
    complete_path = directory / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["files"] = {name: base._file_entry(directory / name, "ExoJAX native atomic preparation")
                         for name in complete["files"]}
    base._write_json(complete_path, complete)


def main(argv=None):
    parser = parser_with_defaults()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.assemble:
            raise ValueError("Assemble a combined manifest separately after auditing atomic opacity assumptions")
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.download_only:
            print(json.dumps(base._file_entry(download_lines(plan), plan["atomic_preparation"]["source_url"])))
            return 0
        if args.chunk_index is not None:
            prepare_chunk(plan, args.chunk_index)
            return 0
        records = completed_chunks(plan)
        complete_path = Path(plan["output_dir"]) / args.species / "complete.json"
        if complete_path.exists() and all(record is not None for record in records):
            complete = json.loads(complete_path.read_text())
            if (complete["plan"]["grid_fingerprint"] != plan["grid_fingerprint"]
                    or complete["plan"].get("atomic_preparation") != plan["atomic_preparation"]):
                raise ValueError("Completed atomic tables use different physics or code; choose a new output directory")
            base._check_files(complete["files"])
            print(f"Reusing completed {args.species} tables", flush=True)
            return 0
        download_lines(plan)
        for index, record in enumerate(records):
            if record is None:
                print(f"Preparing {args.species} chunk {index + 1}/{len(records)}", flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                *arguments, "--chunk-index", str(index)], check=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        merge_species(plan, completed_chunks(plan))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
