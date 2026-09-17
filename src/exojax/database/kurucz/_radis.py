"""Validation and host-array conversion for the optional RADIS Kurucz reader."""

import numpy as np

from exojax.database.core_atom.io import (
    PeriodicTable,
    load_atomicdata,
    load_ionization_energies,
    load_pf_Barklem2016,
    pick_ionE,
)


def validate_species(species):
    """Resolve a supported spectroscopic species before downloading lines."""
    if not isinstance(species, str) or species.count("_") != 1:
        raise ValueError("Use an atomic species such as 'Fe_I' or 'Fe_II'.")
    symbol, stage = species.split("_")
    if symbol not in PeriodicTable or stage not in ("I", "II", "III"):
        raise ValueError(f"Unsupported atomic species {species!r}.")
    ielem = int(np.flatnonzero(PeriodicTable == symbol)[0])
    iion = len(stage)
    _, partition_functions = load_pf_Barklem2016()
    if species not in partition_functions["T[K]"].values:
        raise ValueError(f"No ExoJAX partition function is available for {species}.")
    if ielem not in load_atomicdata()["ielem"].values:
        raise ValueError(f"No ExoJAX atomic metadata is available for {species}.")
    try:
        ion_energy = pick_ionE(ielem, iion, load_ionization_energies())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"No ionization energy is available for {species}.") from exc
    if not np.isfinite(ion_energy) or ion_energy <= 0.0:
        raise ValueError(f"No positive ionization energy is available for {species}.")
    return ielem, iion


def validate_options(nurange, margin, crit, vmr_fraction):
    """Validate the finite spectral interval and optional selection settings."""
    nu = np.asarray(nurange, dtype=float)
    if nu.ndim != 1 or nu.size < 2 or not np.all(np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("nurange must contain at least two finite positive wavenumbers.")
    if np.min(nu) == np.max(nu):
        raise ValueError("nurange must span a nonzero interval.")
    for name, value in (("margin", margin), ("crit", crit)):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if vmr_fraction is not None:
        fractions = np.asarray(vmr_fraction, dtype=float)
        if fractions.shape != (3,) or not np.all(np.isfinite(fractions)) or np.any(
            (fractions < 0.0) | (fractions > 1.0)
        ):
            raise ValueError("vmr_fraction must contain three finite fractions for H, He, H2.")
    return [float(np.min(nu)), float(np.max(nu))]


def transitions_from_dataframe(dataframe, ielem, iion):
    """Convert one RADIS species into sorted, aligned NumPy line arrays.

    RADIS supplies vacuum wavenumbers and level energies in cm-1, Einstein A
    in s-1, and the original logarithmic damping coefficients. Signed Kurucz
    level energies are retained. No air/vacuum conversion is repeated here.
    """
    columns = ("A", "wav", "El", "Eu", "gu", "jl", "ju", "gamRad", "gamSta", "gamvdW")
    missing = set(columns + ("species",)) - set(dataframe.columns)
    if missing:
        raise ValueError(f"RADIS Kurucz data are missing columns: {', '.join(sorted(missing))}.")
    arrays = [dataframe[column].to_numpy(dtype=float) for column in columns]
    for name, values in zip(columns, arrays):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"RADIS Kurucz column {name!r} contains missing or nonfinite values.")
    for name in ("A", "wav", "gu"):
        if np.any(arrays[columns.index(name)] <= 0.0):
            raise ValueError(f"RADIS Kurucz column {name!r} must be positive.")
    if np.any(arrays[5] < 0.0) or np.any(arrays[6] < 0.0):
        raise ValueError("RADIS Kurucz angular momenta must be nonnegative.")
    for code in dataframe["species"].unique():
        try:
            element, charge = (int(part) for part in str(code).split("."))
        except ValueError as exc:
            raise ValueError(f"Invalid RADIS Kurucz species code {code!r}.") from exc
        if (element, charge + 1) != (ielem, iion):
            raise ValueError("RADIS Kurucz data do not match the requested species.")
    order = np.argsort(arrays[1], kind="stable")
    arrays = [values[order] for values in arrays]
    size = len(order)
    return tuple(arrays[:7]) + (
        np.full(size, ielem, dtype=int), np.full(size, iion, dtype=int)
    ) + tuple(arrays[7:])
