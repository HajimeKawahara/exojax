"""Local readers for the Allard alkali density-expansion tables.

Na data: Allard et al. (2019), A&A 628, A120,
https://cdsarc.cds.unistra.fr/ftp/J/A+A/628/A120/ .
The downloaded data remain external to ExoJAX.
"""

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
import re
import tarfile

import numpy as np


@dataclass
class AllardProfile:
    """One temperature's collision profile and optional Na D1 red wing.

    Wavelength is in vacuum Angstrom, density in cm-3, and offsets, width
    (HWHM), and shift in cm-1. ``normalization`` is the supplied pi*r0*f;
    the resulting cross sections, including oscillator strength, are in cm2.
    Coefficient columns start at the first power of density/reference density.
    """

    temperature: float
    wavelength: float
    density: float
    volume: float
    normalization: float
    width: float
    shift: float
    offsets: np.ndarray
    coefficients: np.ndarray
    red_offsets: np.ndarray | None = None
    red_cross_sections: np.ndarray | None = None


def _numbers(text):
    return [float(value.replace("D", "E").replace("d", "e"))
            for value in text.split()]


def _read_profile(text, source="<table>"):
    """Parse a table in memory, allowing its coefficients to wrap lines."""
    try:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        end = next(i for i, line in enumerate(lines) if line.lower() == "end")
        temperature_line = next(line for line in lines[:end]
                                if re.match(r"TK\s*:", line))
        temperature = _numbers(temperature_line.split(":", 1)[1])[0]
        wavelength, density, volume, normalization = _numbers(lines[end + 1])
        count, order = (int(value) for value in lines[end + 2].split())
        width, shift = _numbers(lines[end + 3])
        if count < 2 or order < 1:
            raise ValueError("invalid row count or expansion order")
        values = np.asarray(_numbers(" ".join(lines[end + 4:])))
        expected = count * (order + 1)
        if values.size < expected or values.size % (order + 1):
            raise ValueError("truncated or incomplete coefficient rows")
        # Official Na D1 500/725 K files contain complete surplus rows.
        # Match lect_sig.f by honoring the declared row count.
        data = values[:expected].reshape(count, order + 1)
        scalars = [temperature, wavelength, density, volume,
                   normalization, width, shift]
        if not np.all(np.isfinite(scalars)) or not np.all(np.isfinite(values)):
            raise ValueError("nonfinite table value")
        if min(temperature, wavelength, density, normalization, width) <= 0:
            raise ValueError("nonpositive physical table parameter")
        if volume < 0 or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError("invalid volume or unordered wavenumber offsets")
    except (ValueError, IndexError, StopIteration) as error:
        raise ValueError(f"Invalid Allard profile {source}: {error}") from error
    return AllardProfile(temperature, wavelength, density, volume, normalization,
                         width, shift, data[:, 0], data[:, 1:])


def read_allard_profile(path):
    """Read one local density-expansion table, without interpreting its species.

    The numeric header is authoritative; the descriptive expansion-order text
    is ignored. Negative coefficients are retained for density evaluation.
    """
    path = Path(path)
    return _read_profile(path.read_text(), str(path))


def _iter_files(path):
    """Yield local files, including nested tar members, without extracting them."""
    def unpack(name, data):
        if name.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
                for member in archive:
                    if member.isfile():
                        yield from unpack(f"{name}/{member.name}",
                                          archive.extractfile(member).read())
        elif name.endswith(".omg"):
            yield name, data

    path = Path(path)
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.name.endswith((".omg", ".tar", ".tar.gz", ".tgz")):
                yield from unpack(str(file.relative_to(path)), file.read_bytes())
    else:
        yield from unpack(path.name, path.read_bytes())


def load_allard2019(path):
    """Load Na-H2 D1/D2 tables from the CDS archive or an extracted directory.

    ``path`` is a local ``opacity.tar.gz`` or the extracted ``ALLARD_NaH2``
    directory. Nested archives are read in memory. Both components must have
    matching temperature grids within 500--3000 K and reference density 1e21
    cm-3. Identical example copies are ignored; conflicting copies are rejected.
    """
    tables = {"D1": {}, "D2": {}}
    red_wings = {}
    seen = {}
    for name, content in _iter_files(path):
        basename = name.rsplit("/", 1)[-1]
        if basename.startswith("redwing_NaH2_D1_21_"):
            temperature = float(basename.removesuffix(".omg").rsplit("_", 1)[-1])
            wing = np.loadtxt(BytesIO(content), ndmin=2)
            if (wing.shape[1] != 2 or len(wing) < 2
                    or not np.all(np.isfinite(wing))
                    or np.any(np.diff(wing[:, 0]) <= 0)
                    or np.any(wing[:, 1] < 0)):
                raise ValueError(f"Invalid Na-H2 red wing: {name}")
            if temperature in red_wings and not np.array_equal(red_wings[temperature], wing):
                raise ValueError(f"Conflicting Na-H2 red wings at {temperature} K")
            red_wings[temperature] = wing
            continue
        if not (basename.startswith("table") and "NaH2" in name):
            continue
        components = [component for component in tables if component in name]
        if len(components) != 1:
            raise ValueError(f"Cannot identify Na-H2 doublet component: {name}")
        component = components[0]
        text = content.decode()
        profile = _read_profile(text, name)
        masses = re.search(r"radiator\s+perturber\s+mass:\s*(\S+)\s+(\S+)", text)
        expected_wavelength = {"D1": 5897.558, "D2": 5891.582}[component]
        if (masses is None or not np.allclose(_numbers(" ".join(masses.groups())), [23, 2])
                or not np.isclose(profile.wavelength, expected_wavelength, rtol=0, atol=0.01)
                or profile.density != 1e21
                or not 500 <= profile.temperature <= 3000):
            raise ValueError(f"Not an Allard 2019 Na-H2 {component} table: {name}")
        key = component, profile.temperature
        if key in seen and seen[key] != content:
            raise ValueError(f"Conflicting Na-H2 {component} tables at {profile.temperature} K")
        seen[key] = content
        tables[component][profile.temperature] = profile
    temperatures = sorted(tables["D1"])
    if len(temperatures) < 2 or temperatures != sorted(tables["D2"]):
        raise ValueError("Na-H2 D1 and D2 require matching temperature grids with at least two points")
    if any(temperature not in red_wings for temperature in temperatures):
        raise ValueError("Missing Na-H2 D1 far-red-wing table")
    return {
        "D1": [replace(tables["D1"][temperature],
                       red_offsets=red_wings[temperature][:, 0],
                       red_cross_sections=red_wings[temperature][:, 1])
               for temperature in temperatures],
        "D2": [tables["D2"][temperature] for temperature in temperatures],
    }
