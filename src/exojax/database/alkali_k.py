"""Local reader for the corrected Allard et al. (2024) K--He tables.

Data: https://cdsarc.cds.unistra.fr/ftp/J/A+A/683/A188/ .
The downloaded tables remain external to ExoJAX.
"""

import re

import numpy as np

from exojax.database.alkali import _iter_files, _numbers, _read_profile


def load_allard2024(path):
    """Read K--He D1/D2 tables from a local CDS directory or tar archive.

    All seven temperatures (500, 800, 1000, 1500, 2000, 2500, 3000 K) are
    required for both lines. The corrected D2 table at 800 K is mandatory
    (Allard et al. 2025, A&A 694, C10); the superseded table is ignored.
    Identical example copies are ignored and conflicting copies rejected.
    """
    temperatures = (500, 800, 1000, 1500, 2000, 2500, 3000)
    tables = {"D1": {}, "D2": {}}
    seen = {}
    for name, content in _iter_files(path):
        basename = name.rsplit("/", 1)[-1]
        match = re.fullmatch(r"table(D[12])_KHe_(\d+)_1e21(_2025)?\.omg", basename)
        if match is None:
            continue
        component, temperature, revision = match.groups()
        temperature = int(temperature)
        if (component, temperature) == ("D2", 800) and revision != "_2025":
            continue
        if (temperature not in temperatures
                or (revision and (component, temperature) != ("D2", 800))):
            raise ValueError(f"Unexpected Allard 2024 K--He table: {name}")
        text = content.decode()
        profile = _read_profile(text, name)
        masses = re.search(r"radiator\s+perturber\s+mass:\s*(\S+)\s+(\S+)", text)
        wavelength = {"D1": 7701.1, "D2": 7667.02}[component]
        if (masses is None or not np.allclose(_numbers(" ".join(masses.groups())), [39, 4])
                or not np.isclose(profile.wavelength, wavelength, rtol=0, atol=0.01)
                or profile.density != 1e21 or profile.temperature != temperature):
            raise ValueError(f"Not an Allard 2024 K--He {component} table: {name}")
        key = component, temperature
        if key in seen and seen[key] != content:
            raise ValueError(f"Conflicting K--He {component} tables at {temperature} K")
        seen[key] = content
        tables[component][temperature] = profile
    if 800 not in tables["D2"]:
        raise ValueError(
            "Missing corrected K--He D2 table tableD2_KHe_800_1e21_2025.omg "
            "from Allard et al. (2025), A&A 694, C10."
        )
    if any(sorted(tables[line]) != list(temperatures) for line in tables):
        raise ValueError("Allard 2024 K--He requires all seven temperatures for D1 and D2.")
    return {line: [tables[line][temperature] for temperature in temperatures]
            for line in tables}
