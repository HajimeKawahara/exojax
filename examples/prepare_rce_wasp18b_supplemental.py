"""Build additional molecular CKD/resolved tables on the pinned P7 grid.

This driver preserves the base preparation script and its common grid hash.
Its own version is recorded separately in every chunk and merged archive.
No download or directory creation occurs with --dry-run. Manifest assembly is
deliberately separate: additional opacity and isotope assumptions need review.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import prepare_rce_wasp18b_opacity as base


# Exact species labels from the existing FastChem thermodynamic setup.
SUPPLEMENTAL_LINE_LISTS = {
    "H1O1": "OH/16O-1H/MYTHOS",
    "Fe1H1": "FeH/56Fe-1H/MoLLIST",
    "O1Ti1": "TiO/48Ti-16O/Toto",
    "O1V1": "VO/51V-16O/VOMYT",
    "O1Si1": "SiO/28Si-16O/SiOUVenIR",
}
LINE_LISTS = dict(base.LINE_LISTS, **SUPPLEMENTAL_LINE_LISTS)


def parser_with_defaults():
    parser = base.parser_with_defaults()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest in ("species", "assemble_species"):
            action.choices = sorted(LINE_LISTS)
    parser.set_defaults(species="H1O1")
    return parser


def make_plan(args):
    """Keep the common grid fingerprint and pin this driver separately."""
    args = argparse.Namespace(**vars(args))
    root = args.database_root.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = root / "rce_wasp18b/prepared/supplemental"
    if args.mdb_path is None:
        args.mdb_path = root / LINE_LISTS[args.species]
    plan = base.make_plan(args)
    plan["supplemental_preparation"] = {
        "driver_sha256": base._sha256(Path(__file__).resolve()),
        "base_script_sha256": base._sha256(Path(base.__file__).resolve()),
        "species": args.species,
        "default_line_list": LINE_LISTS[args.species],
    }
    return plan


def prepare_chunk(plan, index):
    """Use the same teacher while adding provenance before table persistence."""
    build_tables = base.build_tables

    def with_provenance(teacher, edges, temperatures, pressure, ng, directory,
                        provenance, resolved_range=None):
        # The CroppedTeacher and the completion record share this dictionary.
        provenance["supplemental_preparation"] = plan["supplemental_preparation"]
        return build_tables(teacher, edges, temperatures, pressure, ng, directory,
                            provenance, resolved_range)

    base.build_tables = with_provenance
    try:
        base.prepare_chunk(plan, index)
    finally:
        base.build_tables = build_tables


def completed_chunks(plan):
    """Reject stale supplemental code as well as changed data or base grids."""
    records = base.completed_chunks(plan)
    for record in records:
        if record is not None and record["provenance"].get("supplemental_preparation") != plan["supplemental_preparation"]:
            raise ValueError("Existing supplemental preparation uses a different driver; choose a new output directory")
    return records


def main(argv=None):
    parser = parser_with_defaults()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.assemble:
            raise ValueError("Assemble a combined manifest separately after auditing additional opacity assumptions.")
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.chunk_index is not None:
            prepare_chunk(plan, args.chunk_index)
            return 0
        records = completed_chunks(plan)
        complete_path = Path(plan["output_dir"]) / args.species / "complete.json"
        if complete_path.exists() and all(record is not None for record in records):
            complete = json.loads(complete_path.read_text())
            if (complete["plan"]["grid_fingerprint"] != plan["grid_fingerprint"]
                    or complete["plan"].get("supplemental_preparation") != plan["supplemental_preparation"]):
                raise ValueError("Completed supplemental species uses a different preparation; choose a new output directory")
            base._check_files(complete["files"])
            print(f"Reusing completed {args.species} tables", flush=True)
            return 0
        for index, record in enumerate(records):
            if record is None:
                print(f"Preparing {args.species} chunk {index + 1}/{len(records)}", flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                *arguments, "--chunk-index", str(index)], check=True)
            else:
                print(f"Reusing {args.species} chunk {index + 1}/{len(records)}", flush=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        base.merge_species(plan, completed_chunks(plan))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
