"""Add verified molecular/atomic preparations to a new WASP-18b manifest.

Keep the existing stellar, CIA and opacity assets. This is an inventory step,
not physical acceptance. Default molecular isotope fractions retain the base
example's explicit approximation that one isotopologue represents its species.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import prepare_rce_wasp18b_opacity as base
from _rce_wasp18b_assets import load_manifest


ISOTOPE_REFERENCE = "https://physics.nist.gov/PhysRefData/Compositions/index.html"
ALIASES = {"H1O1": "OH", "Fe1H1": "FeH", "O1Ti1": "TiO", "O1V1": "VO",
           "O1Si1": "SiO", "Fe1": "Fe", "Na1": "Na", "K1": "K"}


def _absolute_entries(entries, directory):
    return {name: dict(entry, path=str((directory / Path(entry["path"]).expanduser()).resolve()))
            for name, entry in entries.items()}


def read_preparation(path, fingerprint):
    """Verify the merged files, every completed chunk, and its pinned inputs."""
    path = Path(path).expanduser().resolve()
    complete = json.loads(path.read_text())
    plan = complete["plan"]
    if plan["grid_fingerprint"] != fingerprint or base._digest_json(plan["settings"]) != fingerprint:
        raise ValueError(f"Incompatible preparation grid: {path}")
    entries = _absolute_entries(complete["files"], path.parent)
    if not {"ckd.npz", "diffgrid.npz", "diffgrid_metadata.json"} <= set(entries):
        raise ValueError(f"Both broadband and resolved tables are required: {path}")
    base._check_files(entries)
    chunks = _absolute_entries({str(i): entry for i, entry in enumerate(complete["chunk_records"])}, path.parent)
    if len(chunks) != plan["number_of_chunks"]:
        raise ValueError(f"Incomplete chunk inventory: {path}")
    base._check_files(chunks)
    first = None
    for chunk in chunks.values():
        record = json.loads(Path(chunk["path"]).read_text())
        if record["grid_fingerprint"] != fingerprint or record["mdb_path"] != plan["mdb_path"]:
            raise ValueError(f"Incompatible chunk provenance: {chunk['path']}")
        base._check_files(_absolute_entries(record["files"], Path(chunk["path"]).parent))
        base._check_sources(record["provenance"]["sources"])
        for key in ("atomic_preparation", "supplemental_preparation"):
            if record["provenance"].get(key) != plan.get(key):
                raise ValueError(f"Mixed preparation physics or driver versions: {chunk['path']}")
        if first is None:
            first = record["provenance"]
    return complete, entries, first


def isotope_mapping(plan, provenance, fraction=None, fraction_source=None):
    """State the abundance convention without certifying omitted isotopes."""
    atomic = plan.get("atomic_preparation")
    if fraction is not None and (not np.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("Isotope fractions must lie in (0, 1]")
    if fraction is not None and not fraction_source:
        raise ValueError("Explicit isotope fractions require --isotope-source")
    if atomic is not None:
        broadening = atomic["broadening"] + "; perturber fractions H/He/H2=" + str(atomic["perturber_fractions_H_He_H2"])
        convention = "Kurucz published neutral-element line list represents total neutral-element abundance; component/isotope completeness unaudited."
        source = atomic["source_url"]
    else:
        broadening = provenance.get("broadener", "Fixed H2; ExoMol .broad or .def fallback; He/H dependence unaudited.")
        convention = "Dominant isotope represents total bulk species; approximation unaudited. Fraction 1 is not a terrestrial isotopic abundance."
        source = "Existing P7 bulk-species approximation; terrestrial comparison: " + ISOTOPE_REFERENCE
    if fraction is not None:
        convention = "User-specified isotopologue fraction; opacity from other isotopologues is not restored by scaling."
        source = fraction_source
    return dict(isotope_fraction=1. if fraction is None else fraction,
                isotope_convention=convention, isotope_source=source,
                broadening=broadening)


def assemble(manifest_path, preparation_paths, output_path, *, isotope_fractions=None, isotope_source=None):
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output == Path(manifest_path).expanduser().resolve():
        raise ValueError("Choose a new manifest output path; existing manifests are not overwritten")
    baseline = load_manifest(manifest_path)
    config = baseline.config
    config["files"] = {name: dict(entry, path=str(baseline.paths[name]))
                       for name, entry in config["files"].items()}
    fingerprint = config["preparation"]["grid_fingerprint"]
    fractions = {} if isotope_fractions is None else dict(isotope_fractions)
    added = set()
    source_records = []
    for path in preparation_paths:
        path = Path(path).expanduser().resolve()
        complete, files, provenance = read_preparation(path, fingerprint)
        plan, species = complete["plan"], complete["plan"]["species"]
        if species in added or species in config["opacity"]["ckd"]:
            raise ValueError(f"Species already present; select each new preparation once: {species}")
        added.add(species)
        for suffix, filename in (("ckd", "ckd.npz"), ("lbl", "diffgrid.npz"), ("lbl_meta", "diffgrid_metadata.json")):
            key = f"{species}_{suffix}"
            if key in config["files"]:
                raise ValueError(f"Existing asset key would be overwritten: {key}")
            config["files"][key] = files[filename]
        config["files"][f"{species}_preparation"] = base._file_entry(path, "Complete opacity preparation and chunk provenance")
        config["opacity"]["ckd"][species] = dict(asset=f"{species}_ckd", format="npz")
        config["opacity"]["lbl"][species] = dict(asset=f"{species}_lbl", format="diffgrid", metadata_asset=f"{species}_lbl_meta")
        config["opacity"]["mapping"][species] = isotope_mapping(plan, provenance, fractions.get(species), isotope_source)
        config["preparation"]["species"][species] = dict(
            line_list=plan["mdb_path"], no_line_chunks=complete["no_line_chunks"],
            preparation_asset=f"{species}_preparation",
            atomic_preparation=plan.get("atomic_preparation"),
            supplemental_preparation=plan.get("supplemental_preparation"),
            declared_maximum_wavenumber_cm1=provenance.get("declared_maximum_wavenumber_cm1"),
            resolved_coverage_cm1=[plan["settings"]["resolved_nu_min"], plan["settings"]["resolved_nu_max"]],
            line_completeness="Not established by the table temperature range or isotope scaling.",
        )
        source_records.append(base._file_entry(path, "Additional opacity preparation"))
    if set(fractions) - added:
        raise ValueError("Isotope overrides must identify a newly added species")
    included_names = added | {ALIASES.get(species, species) for species in added}
    config["opacity"]["omitted_absorbers"] = [name for name in config["opacity"].get("omitted_absorbers", []) if name not in included_names]
    config["readiness"] = "Additional opacity assets assembled; P7 physical validation is not complete."
    requirements = config.setdefault("missing_requirements", [])
    requirement = "Quantify added/omitted opacity, isotope, broadening and scattering effects on the converged RCE column; assembly does not certify P7."
    if requirement not in requirements:
        requirements.append(requirement)
    config["opacity_assembly"] = dict(
        baseline=base._file_entry(baseline.path, "Input manifest preserved without modification"),
        additional_preparations=source_records,
        driver_sha256=base._sha256(Path(__file__).resolve()), physical_validation_complete=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    base._write_json(output, config)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isotope-fraction", action="append", default=[], metavar="SPECIES=FRACTION")
    parser.add_argument("--isotope-source", help="Source or justification for explicit isotope fractions.")
    args = parser.parse_args(argv)
    try:
        fractions = {}
        for item in args.isotope_fraction:
            species, value = item.split("=", 1)
            if species in fractions:
                raise ValueError(f"Repeated isotope override: {species}")
            fractions[species] = float(value)
        print(assemble(args.manifest, args.preparation, args.output,
                       isotope_fractions=fractions, isotope_source=args.isotope_source))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
