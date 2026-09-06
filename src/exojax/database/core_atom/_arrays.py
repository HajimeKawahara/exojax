"""Shared line-array management for atomic databases."""

import jax.numpy as jnp
import numpy as np

from exojax.database.core_atom.io import (
    load_atomicdata,
    load_ionization_energies,
    pick_ionE,
)


_LINE_FIELDS = (
    "A",
    "elower",
    "eupper",
    "gupper",
    "jlower",
    "jupper",
    "QTmask",
    "ielem",
    "iion",
    "gamRad",
    "gamSta",
    "vdWdamp",
)
_INTEGER_FIELDS = ("QTmask", "ielem", "iion")
_METADATA_FIELDS = ("solarA", "atomicmass", "ionE")


def _set_atomic_metadata(adb):
    """Build NumPy metadata for the selected host line arrays."""
    atomic_data = load_atomicdata().set_index("ielem").loc[adb._ielem]
    adb.solarA = atomic_data["solarA"].to_numpy()
    adb.atomicmass = atomic_data["mass"].to_numpy()
    ionization_energies = load_ionization_energies()
    adb.ionE = np.array(
        [
            pick_ionE(ielem, iion, ionization_energies)
            for ielem, iion in zip(adb._ielem, adb._iion)
        ],
        dtype=float,
    )


def _mask_atomic_lines(
    adb, mask, *, line_fields=_LINE_FIELDS, metadata_fields=_METADATA_FIELDS
):
    """Select host lines and metadata, refreshing existing JAX line arrays."""
    for name in ("nu_lines", "Sij0"):
        setattr(adb, name, getattr(adb, name)[mask])
    for name in line_fields:
        host_name = "_" + name
        setattr(adb, host_name, getattr(adb, host_name)[mask])
    for name in metadata_fields:
        if hasattr(adb, name):
            setattr(adb, name, getattr(adb, name)[mask])
    if hasattr(adb, "dev_nu_lines"):
        _generate_atomic_jnp_arrays(
            adb, line_fields=line_fields, metadata_fields=metadata_fields
        )


def _generate_atomic_jnp_arrays(
    adb, *, line_fields=_LINE_FIELDS, metadata_fields=_METADATA_FIELDS
):
    """Generate JAX arrays from selected lines while preserving fractional J."""
    adb.dev_nu_lines = jnp.array(adb.nu_lines)
    adb.logsij0 = jnp.array(np.log(adb.Sij0))
    for name in line_fields:
        dtype = int if name in _INTEGER_FIELDS else None
        setattr(adb, name, jnp.array(getattr(adb, "_" + name), dtype=dtype))
    for name in metadata_fields:
        setattr(adb, name, jnp.array(getattr(adb, name)))
