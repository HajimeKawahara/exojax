"""Fetch missing raw ExoMol files for the PyExoCross reader.

PyExoCross's downloader currently selects only recommended datasets and keeps
only transition segments enclosed by the requested range. Exact dataset URLs
preserve the dataset and overlapping coverage requested by MdbExomol.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np


def _download_file(url, target, *, optional=False):
    """Stream into a temporary file so interrupted downloads are never reused."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url, timeout=60) as response:
            with NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                first = response.read(1024 * 1024)
                if not first or first.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                    raise ValueError(f"Downloaded empty or HTML content from {url}.")
                if target.suffix == ".bz2" and not first.startswith(b"BZh"):
                    raise ValueError(f"Downloaded file is not bzip2 data: {url}.")
                output.write(first)
                received = len(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and received != int(expected):
                    raise ValueError(f"Incomplete download from {url}.")
            temporary.replace(target)
    except HTTPError as error:
        if not optional or error.code != 404:
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _transition_names(definition, stem, nurange):
    """Select transition segments intersecting the requested wavenumber range."""
    count, maximum = 1, None
    if definition.suffix == ".json":
        transitions = json.loads(definition.read_text())["dataset"]["transitions"]
        count = transitions["number_of_transition_files"]
        maximum = transitions.get("max_wavenumber")
    else:
        for line in definition.read_text().splitlines():
            value, _, comment = line.partition("#")
            if "No. of transition files" in comment:
                count = int(value.split()[0])
            elif "Maximum wavenumber (in cm-1)" in comment:
                maximum = float(value.split()[0])
    if count == 1:
        return [f"{stem}.trans.bz2"]
    if count < 1 or maximum is None or not np.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"Invalid transition-file coverage in {definition}.")
    if stem == "1H-2H-16O__VTT":
        edges = np.array([0, 250, 500, 750, 1000, 1500, 2000, 2250, 2750,
                          3500, 4500, 5500, 7000, 9000, 14000, 20000, 26000])
    else:
        edges = maximum / count * np.arange(count + 1)
    lower, upper = np.min(nurange), np.max(nurange)
    return [
        f"{stem}__{int(left):05d}-{int(right):05d}.trans.bz2"
        for left, right in zip(edges[:-1], edges[1:])
        if right >= lower and left <= upper
    ]


def ensure_exomol_files(path, nurange, bkgdatm="H2", broadf=True, broadf_download=True):
    """Ensure the requested dataset's raw files exist and return its directory.

    ``path`` already includes the caller's ``local_databases`` directory and
    ends in ``molecule/isotopologue/dataset``. Existing text or JSON definitions
    and uncompressed state/transition files are accepted. Missing broadening
    data (HTTP 404) leaves the reader to apply definition-file defaults.
    """
    bounds = np.asarray(nurange, dtype=float)
    if bounds.ndim != 1 or bounds.size < 2 or np.isnan(bounds).any():
        raise ValueError("nurange must contain at least two wavenumbers without NaN.")
    path = Path(path).expanduser().resolve()
    molecule, isotope, dataset = path.parts[-3:]
    stem = f"{isotope}__{dataset}"
    isotope_url = "https://www.exomol.com/db/" + "/".join(
        quote(part, safe="") for part in (molecule, isotope)
    ) + "/"
    dataset_url = isotope_url + quote(dataset, safe="") + "/"
    definition = path / f"{stem}.def.json"
    if not definition.exists():
        definition = path / f"{stem}.def"
        if not definition.exists():
            _download_file(dataset_url + quote(definition.name), definition)

    filenames = [f"{stem}.pf", f"{stem}.states.bz2"]
    filenames += _transition_names(definition, stem, bounds)
    for filename in filenames:
        target = path / filename
        if target.exists() or (target.suffix == ".bz2" and target.with_suffix("").exists()):
            continue
        _download_file(dataset_url + quote(filename), target)

    broad_file = path / f"{isotope}__{bkgdatm}.broad"
    if (broadf and broadf_download and not broad_file.exists()
            and not (path.parent / broad_file.name).exists()):
        _download_file(isotope_url + quote(broad_file.name), broad_file, optional=True)
    return path
