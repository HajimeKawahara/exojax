"""Optional PyExoCross readers for ExoMol-format molecular and atomic lines."""

import numpy as np
import pandas as pd


def import_pyexocross():
    """Import the optional reader dependency and check its supported version."""
    try:
        import pyexocross
    except ModuleNotFoundError as exc:
        if exc.name != "pyexocross":
            raise
        raise ImportError(
            "The PyExoCross reader requires PyExoCross. "
            "Install it with `pip install 'exojax[pyexocross]'`."
        ) from exc
    if pyexocross.__version__ != "1.1.16":
        raise ImportError(
            "The PyExoCross reader supports PyExoCross 1.1.16. "
            "Install it with `pip install 'pyexocross==1.1.16'`."
        )
    return pyexocross


def read_exomol_lines(
    states_file, trans_files, states_columns, nurange,
    extra_state_columns=(), chunk_size=100000,
):
    """Join raw transitions to states without applying database physics.

    The first four state columns are ID, energy, degeneracy and J. Additional
    columns are selected by their explicit positions in ``states_columns``
    and returned with ``_l``/``_u`` suffixes. Unrequested optional columns are
    not parsed. Rows strictly inside the wavenumber bounds are retained.

    A supplied fourth transition column takes precedence over state energy
    differences. If any value is missing, the entire source uses energy
    differences, independently of chunk boundaries and the requested range.
    No partition function, abundance, line strength or broadening is assumed.
    """
    import_pyexocross()
    from pyexocross.base.large_file import read_trans_chunks
    from pyexocross.database.data import textcolumns

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    lower, upper = map(float, nurange)
    if np.isnan([lower, upper]).any() or lower > upper:
        raise ValueError("Invalid wavenumber interval.")
    extras = sorted(set(extra_state_columns), key=list(states_columns).index)
    state_names = ["id", "E", "g", "J"] + extras
    usecols = list(range(4)) + [list(states_columns).index(name) for name in extras]
    columns = [
        "i_upper", "i_lower", "A", "nu_lines", "elower", "eupper",
        "glower", "gup", "jlower", "jupper",
    ] + [f"{name}_{level}" for name in extras for level in ("l", "u")]
    empty = pd.DataFrame({name: pd.Series(dtype=float) for name in columns})
    if not trans_files or lower == upper:
        return empty
    states = pd.concat(
        read_trans_chunks(str(states_file), usecols, state_names, chunk_size),
        ignore_index=True,
    ).set_index("id", verify_integrity=True)
    chunks = []
    for source in trans_files:
        source = str(source)
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
                chunk = chunk[(chunk.nu_lines > lower) & (chunk.nu_lines < upper)].copy()
                if chunk.empty:
                    continue
            upper_states = states.reindex(chunk.i_upper.to_numpy())
            lower_states = states.reindex(chunk.i_lower.to_numpy())
            if upper_states.E.isna().any() or lower_states.E.isna().any():
                raise ValueError(f"Transition references an absent state in {source}.")
            if not supplied_nu:
                chunk["nu_lines"] = upper_states.E.to_numpy() - lower_states.E.to_numpy()
            chunk["elower"] = lower_states.E.to_numpy()
            chunk["eupper"] = upper_states.E.to_numpy()
            chunk["glower"] = lower_states.g.to_numpy()
            chunk["gup"] = upper_states.g.to_numpy()
            chunk["jlower"] = lower_states.J.to_numpy()
            chunk["jupper"] = upper_states.J.to_numpy()
            for name in extras:
                chunk[f"{name}_l"] = lower_states[name].to_numpy()
                chunk[f"{name}_u"] = upper_states[name].to_numpy()
            chunk = chunk[(chunk.nu_lines > lower) & (chunk.nu_lines < upper)].copy()
            if not chunk.empty:
                chunks.append(chunk[columns])
    return pd.concat(chunks, ignore_index=True) if chunks else empty
