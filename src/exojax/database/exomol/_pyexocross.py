"""PyExoCross readers adapted to ExoJAX's ExoMol line-data interface.

Raw reader functions avoid global configuration and range-cache line loss.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import c, h, k

from exojax.database._common.pyexocross import import_pyexocross, read_exomol_lines
from exojax.utils.constants import Tref_original


def _definition(path):
    """Read definition metadata, including optional state-column positions."""
    stem = f"{path.parent.name}__{path.name}"
    json_path = path / f"{stem}.def.json"
    if json_path.exists():
        definition = json.loads(json_path.read_text())
        fields = definition["dataset"]["states"]["states_file_fields"]
        from pyexocross.base.qn_metadata import normalized_states_columns

        columns = normalized_states_columns([field["name"] for field in fields])
        quantum_labels = [
            column
            for field, column in zip(fields[4:], columns[4:])
            if column not in ("unc", "tau", "gfac")
            and not field["name"].startswith("Auxiliary:")
        ]
        broad = definition.get("broad", {})
        return {
            "molmass": float(definition["isotopologue"]["mass_in_Da"]),
            "alpha_ref_def": float(broad.get("default_Lorentzian_half-width", 0.07)),
            "n_Texp_def": float(broad.get("default_temperature_exponent", 0.5)),
            "states_columns": columns,
            "quantum_labels": quantum_labels,
        }

    values = {"alpha_ref_def": 0.07, "n_Texp_def": 0.5}
    scalar_labels = {
        "Isotopologue mass (Da) and (kg)": "molmass",
        "Default value of Lorentzian half-width": "alpha_ref_def",
        "Default value of temperature exponent": "n_Texp_def",
    }
    flags = {"unc": False, "tau": False, "gfac": False}
    flag_labels = {
        "Uncertainty availability": "unc",
        "Lifetime availability": "tau",
        "Lande g-factor availability": "gfac",
    }
    quantum_labels = []
    for line in (path / f"{stem}.def").read_text().splitlines():
        value, _, comment = line.partition("#")
        comment = comment.strip()
        for label, name in scalar_labels.items():
            if comment.startswith(label):
                values[name] = float(value.split()[0])
        for label, name in flag_labels.items():
            if comment.startswith(label):
                flags[name] = bool(int(value.strip()))
        if comment.startswith("Quantum label"):
            quantum_labels.append(value.strip())
    if "molmass" not in values:
        raise ValueError(f"Missing isotopologue mass in {stem}.def")
    values["states_columns"] = (
        ["id", "E", "g", "J"]
        + [name for name, available in flags.items() if available]
        + quantum_labels
    )
    values["quantum_labels"] = quantum_labels
    return values


def _transition_files(path, bounds, preferred_files):
    """Select current raw transition files, also accepting infinite bounds."""
    stem = f"{path.parent.name}__{path.name}"
    pattern = re.compile(re.escape(stem) + r"__(\d+)-(\d+)\.trans(?:\.bz2)?$")
    selected = []
    for filename in preferred_files(str(path), ".trans"):
        name = Path(filename).name
        if name in (f"{stem}.trans", f"{stem}.trans.bz2"):
            selected.append(filename)
            continue
        match = pattern.fullmatch(name)
        if match is not None:
            lower, upper = map(float, match.groups())
            if lower <= bounds[1] and upper >= bounds[0]:
                selected.append(filename)
    return selected


def load_exomol_data(
    path, nurange, optional_quantum_states=False, chunk_size=100000,
    filter_wavenumbers=True,
):
    """Return a pandas line table and metadata from local ExoMol files.

    Only rows strictly inside ``nurange`` are retained. The caller applies
    intensity/energy filters and activation masks. Optional quantum columns
    use the existing ``<label>_l`` and ``<label>_u`` convention. Supported
    transitions have three columns, or a fourth containing line positions.
    A missing fourth-column value makes its entire source use energy
    differences, matching the existing RADIS backend.
    ``filter_wavenumbers=False`` keeps every row of the selected files for
    the public ``nurange=None`` inactive mode.
    """
    import_pyexocross()
    from pyexocross.database.load_exomol import get_statesfile, preferred_files

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    bounds = (
        -np.inf if nurange[0] is None else float(nurange[0]),
        np.inf if nurange[1] is None else float(nurange[1]),
    )
    if np.isnan(bounds).any() or bounds[0] > bounds[1]:
        raise ValueError("Invalid ExoMol wavenumber interval.")
    path = Path(path).expanduser().resolve()
    metadata = _definition(path)
    stem = f"{path.parent.name}__{path.name}"
    partition = np.loadtxt(path / f"{stem}.pf", usecols=(0, 1), ndmin=2)
    metadata["T_gQT"], metadata["gQT"] = partition.T
    qref = np.interp(Tref_original, metadata["T_gQT"], metadata["gQT"])
    sources = _transition_files(path, bounds, preferred_files)
    metadata["trans_file"] = [Path(source) for source in sources]
    if not filter_wavenumbers:
        bounds = (-np.inf, np.inf)
    quantum_labels = metadata["quantum_labels"] if optional_quantum_states else []
    output_columns = [
        "i_upper", "i_lower", "A", "nu_lines", "elower", "eupper",
        "glower", "gup", "jlower", "jupper", "Sij0"
    ] + [f"{label}_{level}" for label in quantum_labels for level in ("l", "u")]
    if not sources or bounds[0] == bounds[1]:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in output_columns}), metadata

    read_path = str(path.parents[2]) + "/"
    data_info = [path.parents[1].name, path.parent.name, path.name]
    states_path = get_statesfile(read_path, data_info)
    lines = read_exomol_lines(
        states_path, sources, metadata["states_columns"], bounds,
        extra_state_columns=quantum_labels, chunk_size=chunk_size,
    )
    # Exact SI constants preserve the reference-strength convention used by
    # RADIS; temperature scaling remains the caller's existing ExoJAX method.
    c2 = h * c / k * 100.0
    lines["Sij0"] = (
        -lines.A * lines.gup
        * np.exp(-c2 * lines.elower / Tref_original)
        * np.expm1(-c2 * lines.nu_lines / Tref_original)
        / (8.0 * np.pi * c * 100.0 * lines.nu_lines**2 * qref)
    )
    return lines[output_columns], metadata


def compute_broadening(
    path, df, bkgdatm="H2", broadf=True, alpha_ref_def=0.07, n_Texp_def=0.5
):
    """Map a0/a1 coefficients to selected rows, retaining definition defaults."""
    alpha = np.full(len(df), alpha_ref_def, dtype=np.float64)
    exponent = np.full(len(df), n_Texp_def, dtype=np.float64)
    if not broadf:
        return alpha, exponent
    path = Path(path)
    filename = f"{path.parent.name}__{bkgdatm}.broad"
    broad_path = next((base / filename for base in (path, path.parent) if (base / filename).exists()), None)
    if broad_path is None:
        warnings.warn(
            f"No local broadening file for {bkgdatm}; using ExoMol definition defaults.",
            UserWarning,
        )
        return alpha, exponent
    import_pyexocross()
    from pyexocross.database.load_exomol import read_broad_file

    broad = read_broad_file(str(broad_path))
    unsupported = set(broad.code) - {"a0", "a1"}
    if unsupported:
        raise NotImplementedError(
            "backend='pyexocross' currently supports only a0/a1 broadening; "
            f"unsupported recipes: {sorted(unsupported)}."
        )
    for recipe, keys in (("a0", ["q1"]), ("a1", ["q1", "q2"])):
        rows = broad[broad.code == recipe]
        if rows.empty:
            continue
        lookup = rows.drop_duplicates(keys, keep="last").set_index(keys)
        target = (
            pd.Index(df.jlower)
            if recipe == "a0"
            else pd.MultiIndex.from_arrays([df.jlower, df.jupper])
        )
        values = lookup[["gamma_L", "n_air"]].reindex(target).to_numpy()
        selected = ~np.isnan(values).any(axis=1)
        alpha[selected], exponent[selected] = values[selected].T
    return alpha, exponent
