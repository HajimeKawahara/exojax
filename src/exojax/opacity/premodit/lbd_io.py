"""Portable HDF5 exchange of prepared ExoMol PreMODIT line densities.

Unlike calculator checkpoints, these files contain no line-sized arrays or
Python objects. Reading requires only NumPy and h5py; writer-only dependencies
are imported inside :func:`save_lbd`.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np

__all__ = ["save_lbd", "load_lbd"]

_FORMAT = "exojax-premodit-lbd"
_VERSION = "1.0"
_ENCODING = "log0_linear_derivatives"
_AXES = "derivative_order,wavenumber,broadening_pair,elower"
_IDENTITY = ("mol_name", "key_iso_ll", "DOI", "Date_ID")
_UNITS = {
    "nu_grid": "cm^-1",
    "elower_grid": "cm^-1",
    "ngamma_ref_grid": "1",
    "n_Texp_grid": "1",
}
_ARRAYS = ("lbd_coeff", *_UNITS, "multi_index_uniqgrid")
_RESERVED = {
    "broadener",
    "valid_wavenumber_range",
    "validation",
    "writer",
    "reference_evaluation",
    "abundance_convention",
    "pressure_convention",
}


def _text(value):
    value = np.asarray(value).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("Expected a UTF-8 string.")
    return value


def _json_copy(value):
    """Copy JSON primitives, rejecting NaN and nonportable Python objects."""
    return json.loads(json.dumps(value, allow_nan=False))


def _validate(arrays, metadata):
    """Validate the numerical contract on both sides of the file boundary."""
    p = metadata["premodit"]
    for name in (*_IDENTITY,):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"{name} must be a nonempty string.")
    if len(metadata["key_iso_ll"].split("__")) != 2:
        raise ValueError("key_iso_ll must have the form isotopologue__linelist.")
    for name in (
        "R",
        "Tref",
        "Twt",
        "Tref_broadening",
        "Tmin",
        "Tmax",
        "dE",
        "c2",
        "doppler_constant",
    ):
        value = p[name]
        if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar.")
    if not np.isfinite(metadata["mol_mass"]) or metadata["mol_mass"] <= 0:
        raise ValueError("mol_mass must be finite and positive.")
    if p["Tmin"] > p["Tmax"]:
        raise ValueError("Tmin must not exceed Tmax.")
    mode = p["diffmode"]
    if not isinstance(mode, (int, np.integer)) or mode not in (0, 1, 2):
        raise ValueError("diffmode must be an integer in {0, 1, 2}.")

    for name in ("lbd_coeff", *_UNITS, "pf_temperature", "pf_Q"):
        a = arrays[name]
        if a.dtype.kind != "f" or a.dtype.itemsize != 8:
            raise ValueError(
                f"{name} must contain float64 data; generate in 64-bit mode."
            )
        if name != "lbd_coeff" and (
            a.ndim != 1 or not a.size or not np.isfinite(a).all()
        ):
            raise ValueError(f"{name} must be a nonempty, finite 1D array.")
    for name in (*_UNITS, "pf_temperature"):
        if np.any(np.diff(arrays[name]) <= 0):
            raise ValueError(f"{name} must be strictly increasing.")
    for name in ("nu_grid", "ngamma_ref_grid", "pf_temperature", "pf_Q"):
        if np.any(arrays[name] <= 0):
            raise ValueError(f"{name} must be positive.")
    nu = arrays["nu_grid"]
    if nu.size < 2 or not np.allclose(
        np.diff(np.log(nu)), 1 / p["R"], rtol=1e-7, atol=1e-12
    ):
        raise ValueError(
            "nu_grid must be equally spaced in log wavenumber and agree with R."
        )
    energies = arrays["elower_grid"]
    if energies.size < 2 or not np.allclose(
        np.diff(energies), p["dE"], rtol=1e-10, atol=1e-10
    ):
        raise ValueError("elower_grid must agree with dE.")
    tq = arrays["pf_temperature"]
    if tq.size < 2 or arrays["pf_Q"].shape != tq.shape:
        raise ValueError(
            "Partition-function temperature and Q arrays must have matching lengths >= 2."
        )
    if not (tq[0] <= min(p["Tref"], p["Tmin"]) and max(p["Tref"], p["Tmax"]) <= tq[-1]):
        raise ValueError(
            "Partition-function table must cover Tref and the design temperature range."
        )

    pairs = arrays["multi_index_uniqgrid"]
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not len(pairs)
        or pairs.dtype.kind not in "iu"
    ):
        raise ValueError(
            "multi_index_uniqgrid must be a nonempty integer array of shape (Nbroad, 2)."
        )
    limits = [len(arrays["ngamma_ref_grid"]), len(arrays["n_Texp_grid"])]
    if np.any(pairs < 0) or np.any(pairs >= limits):
        raise ValueError("Broadening indices are outside their coordinate grids.")
    coeff = arrays["lbd_coeff"]
    if coeff.shape != (mode + 1, len(nu), len(pairs), len(energies)):
        raise ValueError(
            "lbd_coeff shape does not match diffmode and coordinate arrays."
        )
    for order, term in enumerate(coeff):
        if (
            np.isnan(term).any()
            or np.isposinf(term).any()
            or (order > 0 and np.isneginf(term).any())
        ):
            raise ValueError(
                "LBD coefficients must be finite, except for -inf in order zero."
            )

    provenance = metadata["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object.")
    if (
        not isinstance(provenance["broadener"], str)
        or not provenance["broadener"].strip()
    ):
        raise ValueError(
            "broadener must be a nonempty string naming one collision partner."
        )
    bounds = np.asarray(provenance["valid_wavenumber_range"], dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or not nu[0] <= bounds[0] < bounds[1] <= nu[-1]
    ):
        raise ValueError(
            "valid_wavenumber_range must be an increasing pair inside nu_grid."
        )
    if provenance["abundance_convention"] != "isotopologue_no_abundance_weighting":
        raise ValueError("Unsupported isotope abundance convention.")
    if provenance["pressure_convention"] != "broadener_partial_pressure":
        raise ValueError("Unsupported broadener pressure convention.")
    evaluation = provenance["reference_evaluation"]
    if (
        evaluation["nstitch"] != 1
        or evaluation["alias"] != "close"
        or evaluation["profile_kernel"] not in ("analytic", "real_space")
        or evaluation["natural_broadening"] is not False
    ):
        raise ValueError("Unsupported reference evaluation prescription.")
    if not isinstance(provenance["validation"], dict) or not isinstance(
        provenance["writer"], dict
    ):
        raise ValueError("validation and writer must be JSON objects.")


def save_lbd(opa, path, *, metadata):
    """Export an initialized, float64 ExoMol OpaPremodit to a portable .h5 file.

    Supports diffmode 0/1/2 and nstitch=1. ``metadata`` requires ``mol_name``,
    ``key_iso_ll``, ``DOI``, ``Date_ID``, ``broadener``, an explicit
    ``valid_wavenumber_range`` inside the computation grid, and a JSON-compatible
    ``provenance`` dict describing sources and generation choices. Optional
    ``validation`` records separately established accuracy limits.

    Existing paths are never overwritten. Validation and writing complete in a
    temporary sibling file before atomic publication. No backend or line arrays
    are persisted. See ``documents/userguide/lbd_hdf5.rst`` for schema 1.0.
    """
    from exojax import __version__
    from exojax.opacity.premodit.api import OpaPremodit
    from exojax.opacity.providers import ExomolPartitionProvider, ExomolBroadening
    from exojax.utils.constants import hcperk

    path = Path(path)
    if path.suffix not in (".h5", ".hdf5"):
        raise ValueError("An LBD artifact must use .h5 or .hdf5.")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if not isinstance(opa, OpaPremodit):
        raise TypeError("save_lbd expects OpaPremodit.")
    if not getattr(opa, "ready", False):
        raise ValueError("OpaPremodit must be initialized before export.")
    if opa.dbtype != "exomol" or opa.nstitch != 1:
        raise ValueError("LBD schema 1 supports only ExoMol with nstitch=1.")
    if (
        type(opa.pf_provider) is not ExomolPartitionProvider
        or type(opa.broadening_strategy) is not ExomolBroadening
    ):
        raise ValueError(
            "LBD export requires the standard ExoMol partition and broadening providers."
        )
    required = {*_IDENTITY, "broadener", "valid_wavenumber_range", "provenance"}
    if (
        not isinstance(metadata, dict)
        or required - metadata.keys()
        or metadata.keys() - (required | {"validation"})
    ):
        raise ValueError(
            f"metadata requires {sorted(required)}; only validation is optional."
        )
    metadata = _json_copy(metadata)
    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or _RESERVED.intersection(provenance):
        raise ValueError(
            f"provenance must be a dict without reserved keys: {sorted(_RESERVED)}."
        )
    provenance.update(
        broadener=metadata["broadener"],
        valid_wavenumber_range=metadata["valid_wavenumber_range"],
        validation=metadata.get("validation", {}),
        writer={
            "exojax_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        reference_evaluation={
            "profile_kernel": opa.profile_kernel,
            "nstitch": 1,
            "alias": "close",
            "natural_broadening": False,
        },
        abundance_convention="isotopologue_no_abundance_weighting",
        pressure_convention="broadener_partial_pressure",
    )
    pairs, energies, gamma, exponent, resolution, _ = opa._get_info_tuple()
    arrays = dict(
        zip(
            _ARRAYS,
            map(
                np.asarray,
                (
                    opa.lbd_coeff,
                    opa.nu_grid,
                    energies,
                    gamma,
                    exponent,
                    pairs,
                ),
            ),
        )
    )
    arrays.update(
        pf_temperature=np.asarray(opa.pf_provider.T_gQT),
        pf_Q=np.asarray(opa.pf_provider.gQT),
    )
    p = {
        name: float(getattr(opa, name))
        for name in (
            "Tref",
            "Twt",
            "Tref_broadening",
            "Tmin",
            "Tmax",
            "dE",
        )
    }
    p.update(
        R=float(resolution),
        diffmode=opa.diffmode,
        c2=hcperk,
        doppler_constant=3.0415595e-7,
        temperature_unit="K",
        pressure_unit="bar",
        pressure_reference=1.0,
    )
    result = {name: metadata[name] for name in _IDENTITY}
    result.update(mol_mass=float(opa.molmass), premodit=p, provenance=provenance)
    _validate(arrays, result)

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(format=_FORMAT, schema_version=_VERSION)
            for name in _IDENTITY:
                handle.create_dataset(
                    name, data=[result[name]], dtype=h5py.string_dtype("utf-8")
                )
            handle.create_dataset("mol_mass", data=result["mol_mass"]).attrs[
                "units"
            ] = "u"
            handle.create_dataset(
                "provenance_json",
                data=json.dumps(provenance, allow_nan=False),
                dtype=h5py.string_dtype("utf-8"),
            )
            group = handle.create_group("premodit")
            group.attrs.update(p)
            for name in _ARRAYS:
                options = (
                    {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                    if name == "lbd_coeff"
                    else {}
                )
                group.create_dataset(name, data=arrays[name], **options)
            group["lbd_coeff"].attrs.update(encoding=_ENCODING, axes=_AXES)
            group["multi_index_uniqgrid"].attrs["index_base"] = 0
            for name, unit in _UNITS.items():
                group[name].attrs["units"] = unit
            partition = handle.create_group("partition_function")
            partition.attrs["interpolation"] = "linear_Q_in_T"
            partition.create_dataset(
                "temperature", data=arrays["pf_temperature"]
            ).attrs["units"] = "K"
            partition.create_dataset("Q", data=arrays["pf_Q"]).attrs["units"] = "1"
        # A hard link publishes the complete file atomically without clobbering
        # a destination created by another process since the initial check.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_lbd(path):
    """Read and validate portable LBD data as (NumPy arrays, metadata).

    The file is closed before returning. Schema major 1 is accepted, with all
    required fields validated and unknown additive fields ignored. ExoJAX's
    producer version does not constrain the reader. No calculator is rebuilt.
    """
    try:
        with h5py.File(path, "r") as handle:
            version = _text(handle.attrs["schema_version"])
            if (
                _text(handle.attrs["format"]) != _FORMAT
                or re.fullmatch(r"1\.\d+", version) is None
            ):
                raise ValueError("Unsupported PreMODIT LBD format or schema version.")
            group = handle["premodit"]
            partition = handle["partition_function"]
            if (
                _text(group["lbd_coeff"].attrs["encoding"]) != _ENCODING
                or _text(group["lbd_coeff"].attrs["axes"]) != _AXES
            ):
                raise ValueError("Unsupported LBD encoding or axis order.")
            for name, unit in _UNITS.items():
                if _text(group[name].attrs["units"]) != unit:
                    raise ValueError(f"Unsupported units for {name}.")
            if (
                _text(group.attrs["temperature_unit"]),
                _text(group.attrs["pressure_unit"]),
                group.attrs["pressure_reference"],
            ) != ("K", "bar", 1.0):
                raise ValueError(
                    "Expected temperatures in K and reference pressure 1 bar."
                )
            if (
                _text(partition["temperature"].attrs["units"]),
                _text(partition["Q"].attrs["units"]),
                _text(partition.attrs["interpolation"]),
            ) != ("K", "1", "linear_Q_in_T"):
                raise ValueError(
                    "Unsupported partition-function units or interpolation."
                )
            if (
                _text(handle["mol_mass"].attrs["units"]) != "u"
                or group["multi_index_uniqgrid"].attrs["index_base"] != 0
            ):
                raise ValueError(
                    "Expected mass in u and zero-based broadening indices."
                )
            arrays = {name: group[name][()] for name in _ARRAYS}
            arrays.update(
                pf_temperature=partition["temperature"][()], pf_Q=partition["Q"][()]
            )
            metadata = {name: _text(handle[name].asstr()[()]) for name in _IDENTITY}
            metadata.update(
                format=_FORMAT,
                schema_version=version,
                mol_mass=float(handle["mol_mass"][()]),
                premodit=dict(group.attrs),
                partition_function=dict(partition.attrs),
                provenance=json.loads(_text(handle["provenance_json"].asstr()[()])),
            )
        _validate(arrays, metadata)
    except KeyError as exc:
        raise ValueError(f"Missing required LBD field: {exc}") from exc
    return arrays, metadata
