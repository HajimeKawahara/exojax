"""Experimental local ExoMol loader for the PyExoCross comparison script.

This is a proof of the existing MDBSnapshot boundary, not a public backend.
It supports PyExoCross 1.1.16, local text definitions, three/four-column
transitions, and a0/a1 broadening only. It does not download data or build
caches. Run it serially: PyExoCross still changes its own module globals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.database.core.broadening import line_strength_from_Einstein_coeff
from exojax.database.core.line_strength import line_strength_numpy
from exojax.utils.constants import Tref_original, ccgs


def _definition(path):
    """Read the three scalar values needed from a local text definition."""
    labels = {
        "Isotopologue mass (Da) and (kg)": "molmass",
        "Default value of Lorentzian half-width": "alpha_ref",
        "Default value of temperature exponent": "n_Texp",
    }
    values = {}
    for line in path.read_text().splitlines():
        value, _, comment = line.partition("#")
        for label, name in labels.items():
            if label in comment:
                values[name] = float(value.split()[0])
    if values.keys() != {"molmass", "alpha_ref", "n_Texp"}:
        raise ValueError(f"Missing mass or default broadening in {path}")
    return values


def _broadening(path, jlower, jupper, defaults, bkgdatm, broadf):
    """Apply a0, then a1 overrides, retaining definition defaults otherwise."""
    alpha = np.full(len(jlower), defaults["alpha_ref"], dtype=np.float64)
    exponent = np.full(len(jlower), defaults["n_Texp"], dtype=np.float64)
    broad_path = path / f"{path.parent.name}__{bkgdatm}.broad"
    if not broadf or not broad_path.exists():
        return alpha, exponent

    from pyexocross.database.load_exomol import read_broad_file

    broad = read_broad_file(str(broad_path))
    if not set(broad["code"]).issubset({"a0", "a1"}):
        raise ValueError("This experiment supports only a0/a1 broadening recipes.")
    for recipe in ("a0", "a1"):
        for _, row in broad[broad["code"] == recipe].iterrows():
            selected = jlower == row["q1"]
            if recipe == "a1":
                selected &= jupper == row["q2"]
            alpha[selected] = row["gamma_L"]
            exponent[selected] = row["n_air"]
    return alpha, exponent


def load_snapshot(
    path,
    nurange,
    crit=0.0,
    Ttyp=1000.0,
    elower_max=None,
    bkgdatm="H2",
    broadf=True,
    chunk_size=100000,
    *,
    reference_c2=None,
):
    """Return an MDBSnapshot and aligned line diagnostics from local files.

    The path must end in ``molecule/isotopologue/dataset``. Range bounds,
    intensity cutoff at Ttyp, and the lower-energy cutoff are exclusive,
    matching MdbExomol. No isotopic abundance factor is applied. The optional
    reference_c2 overrides the second radiation constant for reference line
    strengths only, to compare physical-constant conventions independently.
    Temperature scaling retains ExoJAX's existing line_strength_numpy.

    PyExoCross's range Parquet cache uses energy differences even when an
    explicit transition wavenumber exists, so this experiment streams the
    original files with cache='none'. A fourth column takes precedence; if
    any value is missing, the entire source uses energy differences, as in
    RADIS. PyExoCross's changes to the caller's __main__ namespace are restored.
    """
    import pyexocross as px
    from pyexocross.base.large_file import read_trans_chunks
    from pyexocross.database.data import textcolumns

    if px.__version__ != "1.1.16":
        raise ValueError("This experiment has been verified with PyExoCross 1.1.16 only.")
    bounds = np.asarray(nurange, dtype=np.float64)
    if bounds.size < 2 or not np.isfinite(bounds).all():
        raise ValueError("This experiment requires a finite spectral range.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    lower, upper = bounds.min(), bounds.max()
    path = Path(path).expanduser().resolve()
    stem = f"{path.parent.name}__{path.name}"
    defaults = _definition(path / f"{stem}.def")
    partition = np.loadtxt(path / f"{stem}.pf", usecols=(0, 1), ndmin=2)
    temperatures, qvalues = partition.T
    qref = np.interp(Tref_original, temperatures, qvalues)
    qtyp = np.interp(Ttyp, temperatures, qvalues)

    # PyExoCross writes configuration names into __main__, including names
    # a comparison driver may already use. This restoration is serial only.
    main_namespace = vars(sys.modules["__main__"])
    original_namespace = main_namespace.copy()
    try:
        data = px.load(
            database="ExoMol",
            molecule=path.parents[1].name,
            isotopologue=path.parent.name,
            dataset=path.name,
            read_path=str(path.parents[2]) + "/",
            save_path=str(path),
            min_range=float(lower),
            max_range=float(upper),
            cache="none",
            chunk_size=chunk_size,
            log="none",
            verbose=False,
        )
    finally:
        for name in main_namespace.keys() - original_namespace.keys():
            del main_namespace[name]
        main_namespace.update(original_namespace)

    states = data.fullstates().set_index("id", verify_integrity=True)
    selected_chunks = []
    transition_rows = 0
    source_columns = {}
    for source in data.transitions:
        ncolumns = textcolumns(source.path)
        if ncolumns not in (3, 4):
            raise ValueError("This experiment supports three/four-column transitions.")
        source_columns[Path(source.path).name] = ncolumns
        names = ["i_upper", "i_lower", "A"]
        use_supplied_nu = ncolumns == 4
        if use_supplied_nu:
            names.append("nu_lines")
            # RADIS falls back for the whole source if any fourth-column
            # value is missing. Scan first so chunk boundaries cannot alter it.
            use_supplied_nu = all(
                chunk["nu_lines"].notna().all()
                for chunk in read_trans_chunks(source, [3], ["nu_lines"], chunk_size)
            )
        for chunk in read_trans_chunks(source, list(range(ncolumns)), names, chunk_size):
            transition_rows += len(chunk)
            upper_states = states.reindex(chunk["i_upper"].to_numpy())
            lower_states = states.reindex(chunk["i_lower"].to_numpy())
            if upper_states["E"].isna().any() or lower_states["E"].isna().any():
                raise ValueError("Transition references a state missing from the states file.")
            if not use_supplied_nu:
                chunk["nu_lines"] = upper_states["E"].to_numpy() - lower_states["E"].to_numpy()
            chunk["elower"] = lower_states["E"].to_numpy()
            chunk["gup"] = upper_states["g"].to_numpy()
            chunk["jlower"] = lower_states["J"].to_numpy(dtype=np.float64)
            chunk["jupper"] = upper_states["J"].to_numpy(dtype=np.float64)
            chunk = chunk[(chunk["nu_lines"] > lower) & (chunk["nu_lines"] < upper)].copy()
            if chunk.empty:
                continue
            chunk["Sref"] = line_strength_from_Einstein_coeff(
                chunk["A"].to_numpy(),
                chunk["gup"].to_numpy(),
                chunk["nu_lines"].to_numpy(),
                chunk["elower"].to_numpy(),
                qref,
            )
            chunk["Sref_exojax"] = chunk["Sref"]
            if reference_c2 is not None:
                chunk["Sref"] = (
                    -chunk["A"] * chunk["gup"]
                    * np.exp(-reference_c2 * chunk["elower"] / Tref_original)
                    * np.expm1(-reference_c2 * chunk["nu_lines"] / Tref_original)
                    / (8.0 * np.pi * ccgs * chunk["nu_lines"] ** 2 * qref)
                )
            selected = line_strength_numpy(
                Ttyp, chunk["Sref"], chunk["nu_lines"], chunk["elower"], qtyp / qref
            ) > crit
            if elower_max is not None:
                selected &= chunk["elower"] < elower_max
            selected_chunks.append(chunk[selected])

    if not selected_chunks or not any(len(chunk) for chunk in selected_chunks):
        raise ValueError("No lines survived the requested filters.")
    lines = pd.concat(selected_chunks, ignore_index=True)
    alpha, exponent = _broadening(
        path, lines["jlower"].to_numpy(), lines["jupper"].to_numpy(), defaults, bkgdatm, broadf
    )
    snapshot = MDBSnapshot(
        meta=MDBMeta("exomol", defaults["molmass"], temperatures, qvalues),
        lines=Lines(
            lines["nu_lines"].to_numpy(),
            lines["elower"].to_numpy(),
            lines["Sref"].to_numpy(),
        ),
        n_Texp=exponent,
        alpha_ref=alpha,
    )
    diagnostics = {
        name: lines[name].to_numpy()
        for name in ("A", "gup", "jlower", "jupper", "i_upper", "i_lower", "Sref_exojax")
    }
    diagnostics.update(
        pyexocross_version=px.__version__,
        transition_rows=transition_rows,
        source_columns=source_columns,
        native_molmass=defaults["molmass"],
        pyexocross_molmass=float(data.config.mass),
    )
    return snapshot, diagnostics
