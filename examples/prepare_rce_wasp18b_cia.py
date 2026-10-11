"""Download and pin CIA tables covering the P7 1500--4500 K domain.

Replace the full H2-H2 table with Borysow, Jorgensen & Fu (2001); do not
blend it with HITRAN. The original inventory is never modified. A manifest
argument must identify a separately assembled working manifest.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np

from _rce_wasp18b_assets import SCHEMA_VERSION, _sha256


BORYSOW_URL = "https://www.astro.ku.dk/~aborysow/programs/CIA.H2H2.Yi"
BORYSOW_REFERENCE = "https://doi.org/10.1016/S0022-4073(00)00023-6"
HITRAN_URL = "https://hitran.org/data/CIA/main/H2-He_2011.cia"
# Borysow (2002), section 2 footnote 1: molecules/cm3 per amagat.
LOSCHMIDT_CM3 = 2.68676e19


def download(url, destination):
    """Stream a missing public asset, publishing only the completed download."""
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty CIA download: {url}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert_borysow(source, destination):
    """Convert the original seven-column amagat table to HITRAN-style cm5.

    The numerical values are divided by LOSCHMIDT_CM3**2. No temperature
    extrapolation, spectral extension, or resampling is performed.
    """
    source, destination = Path(source), Path(destination)
    with source.open() as stream:
        stream.readline()
        temperatures = np.asarray([
            float(value) for value in re.findall(r"([0-9.]+)K", stream.readline())
        ])
    data = np.loadtxt(source, skiprows=3, ndmin=2)
    if (temperatures.size < 2 or data.shape[1] != temperatures.size + 1
            or not np.all(np.isfinite(data)) or np.any(data[:, 0] <= 0)
            or np.any(np.diff(data[:, 0]) <= 0) or np.any(data[:, 1:] < 0)
            or not np.all(np.isfinite(temperatures))
            or np.any(np.diff(temperatures) <= 0)):
        raise ValueError("Invalid Borysow temperature/wavenumber/coefficient table")
    if temperatures[0] > 1500 or temperatures[-1] < 4500:
        raise ValueError("Borysow CIA does not cover 1500--4500 K")
    nu = data[:, 0]
    coefficients = data[:, 1:].T / LOSCHMIDT_CM3**2
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for temperature, coefficient in zip(temperatures, coefficients):
                stream.write(
                    f"{'H2-H2':>20s} {nu[0]:9.3f} {nu[-1]:9.3f} "
                    f"{nu.size:6d} {temperature:7.1f} {coefficient.max():.12E}\n"
                )
                np.savetxt(stream, np.column_stack((nu, coefficient)),
                           fmt=("%12.6f", "%.12E"))
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "temperature_range_K": [float(temperatures[0]), float(temperatures[-1])],
        "temperatures_K": temperatures.tolist(),
        "wavenumber_range_cm1": [float(nu[0]), float(nu[-1])],
        "wavenumber_count": len(nu),
        "input_unit": "cm-1 amagat-2", "output_unit": "cm5 molecule-2",
        "loschmidt_number_cm3": LOSCHMIDT_CM3,
        "unit_reference": "https://www.astro.ku.dk/~aborysow/programs/AApaper.2002.pdf",
        "conversion": "coefficient_cm5 = coefficient_amagat / loschmidt_number_cm3**2",
        "reference": BORYSOW_REFERENCE,
        "selection": "Entire H2-H2 table replaced; no blend with HITRAN2011.",
        "limitations": "Native 1000 K temperature spacing; interpolation error unaudited. "
                       "CIA outside the tabulated spectrum is explicitly set to zero; "
                       "the resulting omitted opacity still requires an error audit.",
    }


def validate_hitran_h2he(path):
    """Check each coefficient and the common native grid before CdbCIA loads it."""
    headers = []
    common_nu = None
    with Path(path).open() as stream:
        while line := stream.readline():
            columns = line.split()
            if not columns:
                continue
            if len(columns) < 5 or columns[0] != "H2-He":
                raise ValueError("Invalid H2-He CIA block header")
            header = (float(columns[1]), float(columns[2]),
                      int(columns[3]), float(columns[4]))
            if not np.all(np.isfinite(header)) or header[2] < 2 or header[3] <= 0:
                raise ValueError("Invalid H2-He CIA block header")
            rows = [stream.readline() for _ in range(header[2])]
            if any(not row.strip() for row in rows):
                raise ValueError("Incomplete H2-He CIA data block")
            data = np.loadtxt(rows, ndmin=2)
            if data.shape != (header[2], 2):
                raise ValueError("Invalid H2-He CIA data block shape")
            nu, coefficient = data.T
            if (not np.all(np.isfinite(data)) or np.any(nu <= 0)
                    or np.any(np.diff(nu) <= 0) or np.any(coefficient < 0)):
                raise ValueError("H2-He CIA requires increasing positive wavenumbers "
                                 "and finite nonnegative coefficients")
            if (nu[0], nu[-1]) != header[:2]:
                raise ValueError("H2-He CIA wavenumber endpoints disagree with the header")
            if common_nu is None:
                common_nu = nu
            elif not np.array_equal(nu, common_nu):
                raise ValueError("H2-He CIA blocks must share the same wavenumber grid")
            headers.append(header)
    temperatures = np.asarray([header[3] for header in headers])
    if (len(headers) < 2 or len({header[:3] for header in headers}) != 1
            or not np.all(np.isfinite(temperatures))
            or np.any(np.diff(temperatures) <= 0)
            or temperatures[0] > 1500 or temperatures[-1] < 4500):
        raise ValueError("H2-He requires a rectangular CIA table covering 1500--4500 K")
    return {
        "temperature_range_K": [float(temperatures[0]), float(temperatures[-1])],
        "wavenumber_range_cm1": list(headers[0][:2]),
        "temperature_count": len(headers), "wavenumber_count": headers[0][2],
    }


def write_json(path, value):
    """Replace a JSON artifact after its complete serialization succeeds."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=Path.home() / ".database")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Update a separate working manifest; never local_inventory.json.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.database_root.expanduser().resolve() / "rce_wasp18b"
    output = (args.output_dir or root / "prepared" / "high_temperature").expanduser().resolve()
    config = None
    if args.manifest is not None:
        args.manifest = args.manifest.expanduser().resolve()
        if args.manifest.name == "local_inventory.json":
            parser.error("Preserve local_inventory.json; pass the separately assembled manifest.")
        config = json.loads(args.manifest.read_text())
        if config.get("schema_version") != SCHEMA_VERSION:
            parser.error(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    raw, converted = output / "CIA.H2H2.Yi", output / "H2-H2_Borysow2001.cia"
    asset_names = {
        pair: config.get("cia", {}).get(pair, {}).get("asset", pair) if config else pair
        for pair in ("H2-H2", "H2-He")
    }
    if len(set(asset_names.values())) != 2:
        parser.error("H2-H2 and H2-He must reference distinct manifest assets.")
    h2he = root / "H2-He_2011.cia"
    previous = None
    if config is not None:
        previous = config.get("files", {}).get(asset_names["H2-He"])
        if previous is not None:
            h2he = (args.manifest.parent / Path(previous["path"]).expanduser()).resolve()
    if not h2he.is_file():
        h2he = output / "H2-He_2011.cia"
    if args.dry_run:
        print(json.dumps({"Borysow_source": BORYSOW_URL, "raw": str(raw),
                          "converted": str(converted), "H2-He": str(h2he),
                          "manifest": str(args.manifest) if args.manifest else None}, indent=2))
        return
    download(BORYSOW_URL, raw)
    report = {"H2-H2": convert_borysow(raw, converted)}
    download(HITRAN_URL, h2he)
    if previous is not None and _sha256(h2he) != previous["sha256"]:
        raise ValueError("H2-He SHA256 differs from the pinned working manifest")
    report["H2-He"] = validate_hitran_h2he(h2he)
    files = {
        asset_names["H2-H2"]: {"path": str(converted), "sha256": _sha256(converted),
                               "source": BORYSOW_URL},
        "H2-H2_Borysow2001_raw": {"path": str(raw), "sha256": _sha256(raw),
                                  "source": BORYSOW_URL},
        asset_names["H2-He"]: {"path": str(h2he), "sha256": _sha256(h2he),
                               "source": HITRAN_URL},
    }
    report["files"] = files
    report_path = output / "cia_preparation.json"
    write_json(report_path, report)
    if config is not None:
        config.setdefault("files", {}).update(files)
        for pair in ("H2-H2", "H2-He"):
            config.setdefault("cia", {})[pair] = {
                "asset": asset_names[pair], "outside_spectral_range": "zero",
                "metadata": report[pair],
            }
        config["files"]["cia_preparation"] = {
            "path": str(report_path), "sha256": _sha256(report_path),
            "source": "examples/prepare_rce_wasp18b_cia.py",
        }
        write_json(args.manifest, config)
    print(f"Prepared CIA tables: {report_path}")


if __name__ == "__main__":
    main()
