"""Build missing ExoMol PyTables caches with bounded transition batches.

Use this before the unchanged WASP-18b opacity job. Existing caches are never
rewritten. Each new cache is published only after all rows and indices finish.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import numpy as np


LINE_LISTS = {
    "H2O1": "H2O/1H2-16O/POKAZATEL",
    "C1O1": "CO/12C-16O/Li2015",
    "C1O2": "CO2/12C-16O2/UCL-4000",
    "H1O1": "OH/16O-1H/MYTHOS",
    "Fe1H1": "FeH/56Fe-1H/MoLLIST",
    "O1Ti1": "TiO/48Ti-16O/Toto",
    "O1V1": "VO/51V-16O/VOMYT",
    "O1Si1": "SiO/28Si-16O/SiOUVenIR",
}


def parser_with_defaults():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database-root", type=Path, default=Path("/home/kawahara/.database"))
    parser.add_argument("--species", choices=sorted(LINE_LISTS), default="C1O2")
    parser.add_argument("--mdb-path", type=Path, help="Override the selected line-list directory.")
    parser.add_argument("--nu-min", type=float, default=200.0)
    parser.add_argument("--nu-max", type=float, default=50000.0)
    parser.add_argument("--rows-per-chunk", type=int, default=250000)
    parser.add_argument("--optional-quantum-states", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_cache(path):
    """Check a cache's table contract without loading its transition rows."""
    import pandas as pd

    with pd.HDFStore(path, mode="r") as store:
        storer = store.get_storer("df")
        if not storer.is_table or storer.nrows < 1 or "nu_lines" not in storer.data_columns:
            raise ValueError(f"Existing cache is not a nonempty searchable transition table: {path}")
        return int(storer.nrows)


def build_transition_cache(raw_path, states, definition, partition_at_reference,
                           *, rows_per_chunk=250000, optional_quantum_states=False):
    """Reproduce RADIS's full-frame calculation while keeping one batch in RAM.

    Uniform three-column files and finite four-column files are supported.
    Mixed missing/present wavenumbers are rejected: RADIS's full-frame fallback
    recomputes every wavenumber when any row is missing, which is not local to a
    batch. Row order and the global CSV index are retained.
    """
    import pandas as pd
    from radis.api.exomolapi import pickup_gE
    from radis.lbl.base import linestrength_from_Einstein

    raw_path = Path(raw_path)
    cache = raw_path.with_suffix(".h5")
    if cache.exists():
        return dict(path=str(cache), rows=validate_cache(cache), reused=True)
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    if not np.isfinite(partition_at_reference) or partition_at_reference <= 0:
        raise ValueError("The partition function at 296 K must be finite and positive")
    string_sizes = {}
    if optional_quantum_states:
        for name in definition["quantum_labels"]:
            if states[name].dtype.kind in "OUS":
                size = int(states[name].astype(str).str.len().max())
                string_sizes.update({f"{name}_l": size, f"{name}_u": size})
    temporary = None
    try:
        with NamedTemporaryFile(dir=cache.parent, prefix=cache.name + ".", suffix=".part", delete=False) as file:
            temporary = Path(file.name)
        rows, missing_wavenumbers, dtypes = 0, None, None
        with pd.HDFStore(temporary, mode="w", complib="blosc", complevel=9) as store:
            with pd.read_csv(
                raw_path, compression="bz2" if raw_path.suffix == ".bz2" else None,
                sep=r"\s+", names=("i_upper", "i_lower", "A", "nu_lines"),
                chunksize=rows_per_chunk,
            ) as reader:
                for trans in reader:
                    if trans.empty:
                        continue
                    missing = trans.nu_lines.isna()
                    if missing.any() and not missing.all():
                        raise ValueError("Mixed missing/present transition wavenumbers require a whole-file fallback")
                    current_missing = bool(missing.all())
                    if missing_wavenumbers is None:
                        missing_wavenumbers = current_missing
                    elif missing_wavenumbers != current_missing:
                        raise ValueError("Mixed missing/present transition wavenumbers across batches are unsupported")
                    if not current_missing and not np.all(np.isfinite(trans.nu_lines)):
                        raise ValueError("Transition wavenumbers must be finite")
                    trans = pickup_gE(
                        states, trans, definition, skip_optional_data=not optional_quantum_states,
                        engine="pytables",
                    )
                    trans["Sij0"] = linestrength_from_Einstein(
                        A=trans["A"], gu=trans["gup"], El=trans["elower"], Ia=1,
                        nu=trans["nu_lines"], Q=partition_at_reference, T=296.0,
                    )
                    if dtypes is None:
                        dtypes = trans.dtypes
                    elif not dtypes.equals(trans.dtypes):
                        raise ValueError("Transition column dtypes changed across batches; cache was not published")
                    store.append("df", trans, format="table", data_columns=["nu_lines"],
                                 index=False, min_itemsize=string_sizes or None)
                    rows += len(trans)
                    print(f"{raw_path.name}: {rows:,} rows cached", flush=True)
                    del trans
            if rows == 0:
                raise ValueError(f"Transition source contains no rows: {raw_path}")
            store.create_table_index("df", columns=["index", "nu_lines"])
            if store.get_storer("df").nrows != rows:
                raise ValueError("Transition cache row count is inconsistent")
            store.flush(fsync=True)
        # Publish atomically without replacing a concurrently created cache.
        try:
            os.link(temporary, cache)
        except FileExistsError:
            return dict(path=str(cache), rows=validate_cache(cache), reused=True)
        return dict(path=str(cache), rows=rows, reused=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dataset_plan(args):
    """Plan without importing RADIS, downloading, or creating directories."""
    from exojax.database.exomol._pyexocross_download import _transition_names

    if args.rows_per_chunk < 1:
        raise ValueError("--rows-per-chunk must be positive")
    if not (np.isfinite(args.nu_min) and np.isfinite(args.nu_max) and 0 < args.nu_min < args.nu_max):
        raise ValueError("Require finite 0 < nu-min < nu-max")
    directory = (args.mdb_path or args.database_root / LINE_LISTS[args.species]).expanduser().resolve()
    molecule, isotope, dataset = directory.parts[-3:]
    stem = f"{isotope}__{dataset}"
    definition = directory / f"{stem}.def"
    names = _transition_names(definition, stem, [args.nu_min, args.nu_max]) if definition.exists() else []
    return dict(
        species=args.species, directory=str(directory), stem=stem,
        url="https://www.exomol.com/db/" + "/".join(quote(part, safe="") for part in (molecule, isotope, dataset)) + "/",
        definition=str(definition), definition_present=definition.exists(),
        rows_per_chunk=args.rows_per_chunk,
        transitions=[dict(raw=str(directory / name), cache=str((directory / name).with_suffix(".h5")),
                          cache_present=(directory / name).with_suffix(".h5").exists()) for name in names],
        missing_definition_note=None if definition.exists() else "Transition names are determined after downloading the definition.",
    )


def prepare(args):
    """Download and cache missing files sequentially with standard RADIS physics."""
    from exojax.database.exomol._pyexocross_download import _download_file
    from radis.api.exomolapi import read_def, read_pf, read_states
    from radis.api.hdf5 import DataFileManager

    plan = dataset_plan(args)
    directory, stem = Path(plan["directory"]), plan["stem"]
    definition_path = Path(plan["definition"])
    if not definition_path.exists():
        _download_file(plan["url"] + quote(definition_path.name), definition_path)
        plan = dataset_plan(args)
    missing = []
    for entry in plan["transitions"]:
        cache = Path(entry["cache"])
        if cache.exists():
            print(f"Reusing {cache}: {validate_cache(cache):,} rows", flush=True)
        else:
            missing.append(entry)
    if not missing:
        return []
    definition = read_def(definition_path)
    pf_path, states_path = directory / f"{stem}.pf", directory / f"{stem}.states.bz2"
    manager = DataFileManager("pytables")
    states_cache = manager.cache_file(states_path)
    if not pf_path.exists():
        _download_file(plan["url"] + quote(pf_path.name), pf_path)
    states = manager.read(states_cache) if states_cache.exists() else None
    required = {"i", "E", "g", "J"}
    if args.optional_quantum_states:
        required.update(definition["quantum_labels"])
    if states is None or not required.issubset(states.columns):
        if not states_path.exists():
            _download_file(plan["url"] + quote(states_path.name), states_path)
        states = read_states(states_path, definition, engine="csv",
                             skip_optional_data=not args.optional_quantum_states)
    pf = read_pf(pf_path)
    partition = np.interp(296.0, pf["T"].to_numpy(), pf["QT"].to_numpy())
    outputs = []
    for entry in missing:
        raw = Path(entry["raw"])
        if not raw.exists():
            _download_file(plan["url"] + quote(raw.name), raw)
        outputs.append(build_transition_cache(
            raw, states, definition, partition, rows_per_chunk=args.rows_per_chunk,
            optional_quantum_states=args.optional_quantum_states,
        ))
    return outputs


def main(argv=None):
    parser = parser_with_defaults()
    args = parser.parse_args(argv)
    try:
        plan = dataset_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            # This job needs host memory only, even when the opacity launcher uses CUDA.
            os.environ["JAX_PLATFORMS"] = "cpu"
            print(json.dumps(prepare(args), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
