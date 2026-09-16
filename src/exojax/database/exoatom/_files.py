"""Resolve raw files for one ExoAtom species and dataset."""

import re
from pathlib import Path
from urllib.parse import quote

from exojax.database.core_atom.io import PeriodicTable
from exojax.database.exomol._pyexocross_download import _download_file


def dataset_identity(path):
    """Return element, charge, and URL path from a canonical dataset path."""
    path = Path(path)
    label, dataset = path.parent.name, path.name
    match = re.fullmatch(r"(\d+)?([A-Z][a-z]?)(_p)?", label)
    if match is None or match[2] not in PeriodicTable:
        raise ValueError("Expected an ExoAtom path such as 'Li/NIST' or 'H/1H/NIST'.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset):
        raise ValueError("Invalid ExoAtom dataset name.")
    atom, charge = match[2], int(match[3] is not None)
    parts = [label, dataset]
    if match[1]:
        element = atom + ("_p" if charge else "")
        if path.parents[1].name != element:
            raise ValueError(f"An isotope path must end in '{element}/{label}/{dataset}'.")
        parts.insert(0, element)
    return atom, charge, parts


def ensure_exoatom_files(path, download=True):
    """Reuse local text/bzip2 files, downloading only missing raw files."""
    path = Path(path)
    _, _, parts = dataset_identity(path)
    stem = f"{path.parent.name}__{path.name}"
    base_url = "https://www.exomol.com/exoatom/db/" + "/".join(
        quote(part, safe="") for part in parts
    ) + "/"
    files = {}
    for extension in ("adef.json", "states", "trans", "pf"):
        target = path / f"{stem}.{extension}"
        compressed = target.with_name(target.name + ".bz2")
        if extension in ("states", "trans") and compressed.is_file():
            target = compressed
        elif not target.is_file():
            if not download:
                raise FileNotFoundError(f"Missing ExoAtom file: {target}")
            _download_file(base_url + quote(target.name), target)
        files[extension] = target
    return files
