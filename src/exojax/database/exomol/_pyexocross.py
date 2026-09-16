"""PyExoCross readers adapted to ExoJAX's ExoMol line-data interface.

Only the reader functions are used: the high-level PyExoCross loader changes
process configuration and its range caches discard supplied line positions.
Raw files are streamed without creating a backend cache.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import c, h, k

from exojax.utils.constants import Tref_original


def import_pyexocross():
    """Import the optional reader dependency and check its supported version."""
    try:
        import pyexocross
    except ModuleNotFoundError as exc:
        if exc.name != "pyexocross":
            raise
        raise ImportError(
            "backend='pyexocross' requires PyExoCross. "
            "Install it with `pip install 'exojax[pyexocross]'`."
        ) from exc
    if pyexocross.__version__ != "1.1.16":
        raise ImportError(
            "backend='pyexocross' supports PyExoCross 1.1.16. "
            "Install it with `pip install 'pyexocross==1.1.16'`."
        )
    return pyexocross


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
    from pyexocross.base.large_file import read_trans_chunks
    from pyexocross.database.data import textcolumns
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
        "i_upper", "i_lower", "A", "nu_lines", "elower", "gup", "jlower", "jupper", "Sij0"
    ] + [f"{label}_{level}" for label in quantum_labels for level in ("l", "u")]
    if not sources or bounds[0] == bounds[1]:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in output_columns}), metadata

    read_path = str(path.parents[2]) + "/"
    data_info = [path.parents[1].name, path.parent.name, path.name]
    states_path = get_statesfile(read_path, data_info)
    state_columns = metadata["states_columns"] if optional_quantum_states else ["id", "E", "g", "J"]
    states = pd.concat(
        read_trans_chunks(states_path, list(range(len(state_columns))), state_columns, chunk_size),
        ignore_index=True,
    ).set_index("id", verify_integrity=True)
    chunks = []
    # Exact SI constants preserve the reference-strength convention used by
    # RADIS; temperature scaling remains the caller's existing ExoJAX method.
    c2 = h * c / k * 100.0
    for source in sources:
        ncolumns = textcolumns(source)
        if ncolumns not in (3, 4):
            raise ValueError(f"Expected three/four transition columns in {source}.")
        names = ["i_upper", "i_lower", "A"] + (["nu_lines"] if ncolumns == 4 else [])
        supplied_nu = ncolumns == 4 and all(
            chunk["nu_lines"].notna().all()
            for chunk in read_trans_chunks(source, [3], ["nu_lines"], chunk_size)
        )
        for chunk in read_trans_chunks(source, list(range(ncolumns)), names, chunk_size):
            if supplied_nu:
                chunk = chunk[(chunk.nu_lines > bounds[0]) & (chunk.nu_lines < bounds[1])].copy()
                if chunk.empty:
                    continue
            upper_states = states.reindex(chunk.i_upper.to_numpy())
            lower_states = states.reindex(chunk.i_lower.to_numpy())
            if upper_states.E.isna().any() or lower_states.E.isna().any():
                raise ValueError(f"Transition references an absent state in {source}.")
            if not supplied_nu:
                chunk["nu_lines"] = upper_states.E.to_numpy() - lower_states.E.to_numpy()
            chunk["elower"] = lower_states.E.to_numpy()
            chunk["gup"] = upper_states.g.to_numpy()
            chunk["jlower"] = lower_states.J.to_numpy()
            chunk["jupper"] = upper_states.J.to_numpy()
            for label in quantum_labels:
                chunk[f"{label}_l"] = lower_states[label].to_numpy()
                chunk[f"{label}_u"] = upper_states[label].to_numpy()
            chunk = chunk[(chunk.nu_lines > bounds[0]) & (chunk.nu_lines < bounds[1])].copy()
            if chunk.empty:
                continue
            chunk["Sij0"] = (
                -chunk.A * chunk.gup
                * np.exp(-c2 * chunk.elower / Tref_original)
                * np.expm1(-c2 * chunk.nu_lines / Tref_original)
                / (8.0 * np.pi * c * 100.0 * chunk.nu_lines**2 * qref)
            )
            chunks.append(chunk[output_columns])
    if not chunks:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in output_columns}), metadata
    return pd.concat(chunks, ignore_index=True), metadata


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
