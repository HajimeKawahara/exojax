r"""Export local ExoMol data as LBD HDF5, reference spectra and a manifest.

Install the optional exojax[pyexocross] environment first. This example requires
the raw files to be present locally and does not download or publish data.
Start with a small band: increasing grid size and line density increases RAM.

Example:
    python export_premodit_lbd.py /data/CO/12C-16O/Li2015 ./co-lbd \
        --nu-range 4200 4300 --valid-range 4220 4280 \
        --temperature-range 800 1200 --doi 10.1088/0067-0049/216/1/15 \
        --release pilot-1

The valid range is a requested inner band, not an accuracy certification.
Enlarge the guard band and compare spectra before publishing a validated band.
Reference spectra test reader/kernel agreement; they are not a direct-line
accuracy benchmark. Pressures are the single broadener's partial pressure.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def fingerprint(path):
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require_local_files(path, bounds):
    """Check the same raw files that MdbExomol would otherwise download."""
    from exojax.database.exomol._pyexocross_download import _transition_names

    stem = f"{path.parent.name}__{path.name}"
    definition = path / f"{stem}.def.json"
    if not definition.is_file():
        definition = path / f"{stem}.def"
    if not definition.is_file():
        raise FileNotFoundError(f"Download the ExoMol definition first: {definition}")
    names = [f"{stem}.pf", f"{stem}.states.bz2"]
    names += _transition_names(definition, stem, bounds)
    for name in names:
        candidate = path / name
        if candidate.is_file():
            continue
        if candidate.suffix == ".bz2" and candidate.with_suffix("").is_file():
            continue
        raise FileNotFoundError(f"Download the raw ExoMol file first: {candidate}")
    return definition


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mdb_path", type=Path, help="Local molecule/isotopologue/linelist directory")
    parser.add_argument("output", type=Path, help="Directory for the three output files")
    parser.add_argument("--nu-range", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--valid-range", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--temperature-range", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--points", type=int, default=4096, help="Even ESLOG grid size (default: 4096)")
    parser.add_argument("--diffmode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--broadener", default="H2")
    parser.add_argument("--pressures", nargs="+", type=float, default=[0.01, 1.0, 10.0], help="Reference pressures in bar")
    parser.add_argument("--doi", required=True, help="Source line-list DOI")
    parser.add_argument("--release", required=True, help="LBD data release identifier")
    args = parser.parse_args()
    lo, hi = args.nu_range
    vlow, vhigh = args.valid_range
    tmin, tmax = args.temperature_range
    if not (0 < lo < vlow < vhigh < hi and 0 < tmin < tmax):
        parser.error("Use positive increasing ranges with the valid band strictly inside the grid.")
    if args.points < 4 or args.points % 2 or not np.isfinite([lo, hi, tmin, tmax]).all():
        parser.error("Use finite ranges and an even grid size of at least four.")
    if not np.isfinite(args.pressures).all() or np.any(np.asarray(args.pressures) <= 0):
        parser.error("Reference pressures must be finite and positive.")

    from jax import config
    config.update("jax_enable_x64", True)
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit, save_lbd

    path = args.mdb_path.expanduser().resolve()
    definition = require_local_files(path, args.nu_range)
    molecule, isotope, linelist = path.parts[-3:]
    stem = f"{isotope}__{linelist}.{args.broadener}"
    output = args.output.expanduser().resolve()
    targets = [output / f"{stem}.premodit.h5", output / f"{stem}.reference.npz", output / "manifest.json"]
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
    nu_grid = np.geomspace(lo, hi, args.points)
    mdb = MdbExomol(path, nu_grid, backend="pyexocross", gpu_transfer=False,
                   bkgdatm=args.broadener, broadf_download=False, crit=0.0)
    from pyexocross.database.load_exomol import get_statesfile

    state_path = Path(get_statesfile(str(path.parents[2]) + "/", [molecule, isotope, linelist]))
    broad_name = f"{isotope}__{args.broadener}.broad"
    broad_path = next((base / broad_name for base in (path, path.parent)
                       if (base / broad_name).is_file()), None)
    sources = [definition, path / f"{isotope}__{linelist}.pf", state_path, *mdb.trans_file]
    if broad_path is not None:
        sources.append(broad_path)
    provenance = {
        "isotopologue": isotope, "linelist": linelist,
        "source_url": f"https://www.exomol.com/db/{molecule}/{isotope}/{linelist}/",
        "source_files": [fingerprint(source) for source in sources],
        "loader": {"name": "pyexocross", "version": version("pyexocross")},
        "molecular_mass_source": definition.name,
        "premodit_settings": {"auto_trange_K": args.temperature_range, "diffmode": args.diffmode,
                              "broadening_resolution": {"mode": "manual", "value": 0.2}},
        "line_selection": {"crit": 0.0, "crit_unit": "cm/molecule", "Ttyp": float(mdb.Ttyp),
                           "elower_max": None, "wavenumber_range": args.nu_range},
        "broadening": {"file": broad_path.name if broad_path else None,
                       "default_hwhm_cm-1_per_bar": float(mdb.alpha_ref_def),
                       "default_temperature_exponent": float(mdb.n_Texp_def),
                       "fallback": "all_lines" if broad_path is None else "unmatched_lines_count_not_recorded"},
        "line_position_rule": "complete_fourth_column_else_state_energy_difference_per_file",
    }
    opa = OpaPremodit.from_snapshot(mdb.to_snapshot(), nu_grid,
                                   auto_trange=(tmin, tmax), diffmode=args.diffmode,
                                   broadening_resolution={"mode": "manual", "value": 0.2},
                                   nstitch=1, profile_kernel="analytic")
    temperatures = np.array([tmin, np.sqrt(tmin * tmax), tmax])
    t_eval = np.repeat(temperatures, len(args.pressures))
    p_eval = np.tile(args.pressures, len(temperatures))
    reference = np.asarray(opa.xsmatrix(t_eval, p_eval))
    output.mkdir(parents=True, exist_ok=True)
    save_lbd(opa, targets[0], metadata={
        "mol_name": molecule, "key_iso_ll": f"{isotope}__{linelist}",
        "DOI": args.doi, "Date_ID": args.release, "broadener": args.broadener,
        "valid_wavenumber_range": args.valid_range, "provenance": provenance,
        "validation": {"status": "reference_only", "guard_convergence_checked": False,
                       "temperatures_K": temperatures.tolist(), "pressures_bar": args.pressures},
    })
    with targets[1].open("xb") as handle:
        np.savez_compressed(handle, nu_grid=nu_grid, temperature_K=t_eval, pressure_bar=p_eval,
                            cross_section_cm2_per_molecule=reference, profile_kernel="analytic")
    manifest = {
        "format": "exojax-premodit-lbd", "schema_version": "1.0", "release": args.release,
        "mol_name": molecule, "key_iso_ll": f"{isotope}__{linelist}", "broadener": args.broadener,
        "wavenumber_range_cm-1": args.nu_range, "valid_wavenumber_range_cm-1": args.valid_range,
        "temperature_range_K": args.temperature_range, "R": float(opa.resolution),
        "diffmode": args.diffmode, "validation_status": "reference_only",
        "files": [fingerprint(target) for target in targets[:2]],
    }
    with targets[2].open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {targets[0]}, {targets[1]} and {targets[2]}")


if __name__ == "__main__":
    main()
