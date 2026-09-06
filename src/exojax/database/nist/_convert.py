"""Normalize RADIS NIST species and line columns."""

import re
import warnings

import numpy as np

from exojax.database.core_atom.io import PeriodicTable


_COLUMNS = {
    "nu_lines": "wav", "A": "A", "elower": "El", "eupper": "Eu",
    "glower": "gl", "gupper": "gu", "jlower": "jl", "jupper": "ju",
}


def parse_species(species):
    """Return canonical RADIS notation, atomic number, and ion stage."""
    if not isinstance(species, str):
        raise ValueError("species must be an atom and ion stage, such as 'Fe_II'.")
    canonical = "_".join(species.split())
    match = re.fullmatch(r"([A-Z][a-z]?)_([IVXLC]+)", canonical)
    if match is None or match[1] not in PeriodicTable:
        raise ValueError("species must be an atom and ion stage, such as 'Fe_II'.")
    symbol, roman = match.groups()
    if roman not in ("I", "II", "III"):
        raise ValueError("Only ion stages I, II, and III have supported partition functions.")
    stage = len(roman)
    ielem = int(np.flatnonzero(PeriodicTable == symbol)[0])
    if stage > ielem + 1:
        raise ValueError(f"Invalid ion stage in species '{species}'.")
    return canonical, ielem, stage


def line_arrays(dataframe, species):
    """Select finite, physical line records and sort their aligned columns."""
    missing = (set(_COLUMNS.values()) | {"species"}) - set(dataframe.columns)
    if missing:
        raise ValueError(f"NIST data are missing required columns: {', '.join(sorted(missing))}.")
    if not np.all(dataframe["species"].to_numpy() == species):
        raise ValueError(f"NIST data contain species other than the requested '{species}'.")
    arrays = {name: dataframe[column].to_numpy(dtype=float)
              for name, column in _COLUMNS.items()}
    valid = np.isfinite(np.stack(list(arrays.values()))).all(axis=0)
    for name in ("nu_lines", "A", "glower", "gupper"):
        valid &= arrays[name] > 0
    for name in ("elower", "jlower", "jupper"):
        valid &= arrays[name] >= 0
    valid &= arrays["eupper"] > arrays["elower"]
    if not np.all(valid):
        warnings.warn(
            f"Discarded {np.count_nonzero(~valid)} NIST lines with missing or invalid parameters.",
            UserWarning,
            stacklevel=2,
        )
    order = np.argsort(arrays["nu_lines"][valid], kind="stable")
    return {name: values[valid][order] for name, values in arrays.items()}
