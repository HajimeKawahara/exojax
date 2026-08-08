"""Persistence helpers for pressure-layer aligned Diffgrid tables.

The ``diffgrid@1`` format stores the arrays in ``REQUIRED_DIFFGRID_ARRAYS``
plus optional ``wav``. NPZ artifacts use a sibling ``*_metadata.json`` file;
Zarr artifacts store the same metadata as group attributes. Metadata declares
every array's shape, dtype, and SHA-256 digest, where the digest covers the
C-order bytes followed by the dtype and shape strings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, TYPE_CHECKING

import jax
import jaxlib
import numpy as np

from exojax import __version__
from exojax.opacity.io.serialization import SCHEMA_OPA_VERSION
from exojax.opacity.io.serialization import _ensure_compatible_versions
from exojax.opacity.io.serialization import _load
from exojax.opacity.io.serialization import _load_from_npz
from exojax.opacity.io.serialization import _load_from_zarr
from exojax.opacity.io.serialization import _sha256_array
from exojax.opacity.io.serialization import _validate_schema

if TYPE_CHECKING:
    from exojax.opacity.diffgrid.api import OpaDiffgrid


DIFFGRID_SCHEMA_VERSION = "diffgrid@1"

REQUIRED_DIFFGRID_ARRAYS = (
    "nu_grid",
    "pressure_grid",
    "temperature_grid",
    "inverse_temperature_grid",
    "log_cross_section_grid",
    "log_cross_section_derivative_grid",
    "log_cross_section_floor",
)

_OPTIONAL_SCALAR_ATTRIBUTES = (
    "wavelength_order",
    "resolution",
    "molmass",
)

_UNITS = {
    "wavenumber": "cm^-1",
    "pressure": "bar",
    "temperature": "K",
    "inverse_temperature": "K^-1",
    "cross_section": "cm^2",
    "log_cross_section_derivative": "K",
}


def _normalize_json_value(value: Any) -> Any:
    """Convert a value to JSON-compatible Python primitives."""
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, np.ndarray) or isinstance(value, jax.Array):
        return _normalize_json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _normalize_json_value(value.item())
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON metadata numbers must be finite.")
        return value
    raise TypeError(
        "Metadata contains unsupported type "
        f"{type(value).__name__}. Only JSON-serializable primitives, lists, "
        "dicts, and NumPy or JAX scalars/arrays are allowed."
    )


def _normalize_json_mapping(name: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a metadata mapping and provide a focused type error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): _normalize_json_value(item) for key, item in value.items()}


def _diffgrid_arrays(opa: OpaDiffgrid) -> Dict[str, np.ndarray]:
    """Collect the self-contained Diffgrid payload from an initialized object."""
    info = opa.diffgrid_info
    arrays = {
        "nu_grid": np.asarray(opa.nu_grid),
        "pressure_grid": np.asarray(info.pressure_grid),
        "temperature_grid": np.asarray(info.temperature_grid),
        "inverse_temperature_grid": np.asarray(info.inverse_temperature_grid),
        "log_cross_section_grid": np.asarray(info.log_cross_section_grid),
        "log_cross_section_derivative_grid": np.asarray(
            info.log_cross_section_derivative_grid
        ),
        "log_cross_section_floor": np.asarray(info.log_cross_section_floor),
    }
    if hasattr(opa, "wav"):
        arrays["wav"] = np.asarray(opa.wav)
    return arrays


def _diffgrid_metadata(
    opa: OpaDiffgrid,
    arrays: Dict[str, np.ndarray],
    extra_meta: Mapping[str, Any] | None,
    aux: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Build portable metadata for a Diffgrid archive."""
    optional_attributes = {}
    for attribute in _OPTIONAL_SCALAR_ATTRIBUTES:
        if hasattr(opa, attribute):
            optional_attributes[attribute] = _normalize_json_value(
                getattr(opa, attribute)
            )

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_OPA_VERSION,
        "diffgrid_schema_version": DIFFGRID_SCHEMA_VERSION,
        "opa_type": "OpaDiffgrid",
        "exojax_version": __version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": dict(_UNITS),
        "array_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "array_dtypes": {
            name: str(array.dtype) for name, array in arrays.items()
        },
        "array_digests": {
            name: _sha256_array(array) for name, array in arrays.items()
        },
        "opa_state": {
            "teacher_method": _normalize_json_value(
                getattr(opa, "teacher_method", None)
            ),
            "optional_attributes": optional_attributes,
        },
    }
    if extra_meta is not None:
        meta["user_meta"] = _normalize_json_mapping("extra_meta", extra_meta)
    if aux is not None:
        meta["aux"] = _normalize_json_mapping("aux", aux)
    return meta


def _npz_paths(path: str) -> Tuple[Path, Path]:
    """Resolve the NPZ payload and sidecar metadata paths."""
    path_string = str(path)
    npz_path = Path(path_string if path_string.endswith(".npz") else path_string + ".npz")
    metadata_path = npz_path.with_name(npz_path.stem + "_metadata.json")
    return npz_path, metadata_path


def _save_as_npz(
    path: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]
) -> None:
    """Write a compressed NPZ payload and its JSON metadata sidecar."""
    npz_path, metadata_path = _npz_paths(path)
    np.savez_compressed(npz_path, **arrays)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=4, sort_keys=True, allow_nan=False)


def _save_as_zarr(
    path: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]
) -> None:
    """Write a Zarr v2 or v3 Diffgrid archive."""
    import zarr

    try:
        major_version = int(zarr.__version__.split(".")[0])
    except (AttributeError, TypeError, ValueError):
        major_version = 2

    path_string = str(path)
    zarr_path = path_string if path_string.endswith(".zarr") else path_string + ".zarr"
    group = zarr.open(zarr_path, mode="w")
    if major_version >= 3:
        from zarr import codecs

        compressor = codecs.GzipCodec(level=1)
        for name, array in arrays.items():
            dataset = group.create_array(
                name,
                shape=array.shape,
                dtype=array.dtype,
                compressors=[compressor],
            )
            dataset[...] = array
    else:
        compressor = zarr.get_codec({"id": "zlib", "level": 1})
        for name, array in arrays.items():
            group.create_dataset(name, data=array, compressor=compressor)

    group.attrs.update(meta)
    close_group = getattr(group, "close", None)
    if callable(close_group):
        close_group()
    else:
        close_store = getattr(getattr(group, "store", None), "close", None)
        if callable(close_store):
            close_store()


def _is_real_numeric(array: np.ndarray) -> bool:
    """Return whether an array uses a non-boolean real numeric dtype."""
    return np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.floating
    )


def _validate_coordinate(
    name: str, array: np.ndarray, *, minimum_size: int
) -> None:
    """Validate a positive, finite, one-dimensional coordinate array."""
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values.")
    if not _is_real_numeric(array):
        raise ValueError(f"{name} must use a real numeric dtype.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} values must be finite and positive.")


def _coordinate_rtol(*arrays: np.ndarray) -> float:
    """Choose a reciprocal-coordinate tolerance from the stored dtypes."""
    epsilons = [
        float(np.finfo(array.dtype).eps)
        for array in arrays
        if np.issubdtype(array.dtype, np.floating)
    ]
    return 8.0 * max(epsilons, default=np.finfo(np.float64).eps)


def _validate_array_values(arrays: Dict[str, np.ndarray]) -> None:
    """Validate Diffgrid arrays without converting them to active JAX dtypes."""
    missing = [name for name in REQUIRED_DIFFGRID_ARRAYS if name not in arrays]
    if missing:
        raise KeyError(f"Missing required Diffgrid arrays: {', '.join(missing)}")

    for name in tuple(arrays):
        arrays[name] = np.asarray(arrays[name])

    nu_grid = arrays["nu_grid"]
    pressure_grid = arrays["pressure_grid"]
    temperature_grid = arrays["temperature_grid"]
    inverse_temperature_grid = arrays["inverse_temperature_grid"]
    log_cross_section_grid = arrays["log_cross_section_grid"]
    derivative_grid = arrays["log_cross_section_derivative_grid"]
    log_cross_section_floor = arrays["log_cross_section_floor"]

    _validate_coordinate("nu_grid", nu_grid, minimum_size=1)
    _validate_coordinate("pressure_grid", pressure_grid, minimum_size=1)
    _validate_coordinate("temperature_grid", temperature_grid, minimum_size=2)
    _validate_coordinate(
        "inverse_temperature_grid", inverse_temperature_grid, minimum_size=2
    )
    if np.unique(nu_grid).size != nu_grid.size:
        raise ValueError("nu_grid values must be distinct.")

    comparison_dtype = np.result_type(
        temperature_grid.dtype, inverse_temperature_grid.dtype, np.float64
    )
    temperature_for_comparison = temperature_grid.astype(
        comparison_dtype, copy=False
    )
    inverse_for_comparison = inverse_temperature_grid.astype(
        comparison_dtype, copy=False
    )
    if np.any(np.diff(inverse_for_comparison) <= 0.0):
        raise ValueError("inverse_temperature_grid must be strictly increasing.")
    if np.any(np.diff(temperature_for_comparison) >= 0.0):
        raise ValueError(
            "temperature_grid must be strictly decreasing to match "
            "inverse_temperature_grid."
        )
    expected_inverse_temperature = 1.0 / temperature_for_comparison
    if not np.allclose(
        inverse_for_comparison,
        expected_inverse_temperature,
        rtol=_coordinate_rtol(temperature_grid, inverse_temperature_grid),
        atol=0.0,
    ):
        raise ValueError(
            "temperature_grid and inverse_temperature_grid are inconsistent."
        )

    for name, array in (
        ("log_cross_section_grid", log_cross_section_grid),
        ("log_cross_section_derivative_grid", derivative_grid),
    ):
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must use a floating dtype.")
        if array.ndim != 3:
            raise ValueError(f"{name} must be three-dimensional.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    expected_shape = (
        pressure_grid.size,
        temperature_grid.size,
        nu_grid.size,
    )
    if log_cross_section_grid.shape != expected_shape:
        raise ValueError(
            "log_cross_section_grid shape does not match the Diffgrid "
            f"coordinates: expected {expected_shape}, got "
            f"{log_cross_section_grid.shape}."
        )
    if derivative_grid.shape != expected_shape:
        raise ValueError(
            "log_cross_section_derivative_grid shape does not match the "
            f"Diffgrid coordinates: expected {expected_shape}, got "
            f"{derivative_grid.shape}."
        )

    if log_cross_section_floor.ndim != 0:
        raise ValueError("log_cross_section_floor must be a scalar.")
    if not np.issubdtype(log_cross_section_floor.dtype, np.floating):
        raise ValueError("log_cross_section_floor must use a floating dtype.")
    if not bool(np.isfinite(log_cross_section_floor)):
        raise ValueError("log_cross_section_floor must be finite.")

    if "wav" in arrays:
        wav = arrays["wav"]
        _validate_coordinate("wav", wav, minimum_size=nu_grid.size)
        if wav.shape != nu_grid.shape:
            raise ValueError("wav shape must match nu_grid.")


def _require_metadata(meta: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    """Require metadata keys with a corruption-oriented error message."""
    missing = [key for key in keys if key not in meta]
    if missing:
        raise KeyError(f"Missing required Diffgrid metadata: {', '.join(missing)}")


def _validate_archive_metadata(
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    *,
    strict: bool,
    allow_downgrade: bool,
) -> None:
    """Validate schemas, provenance, array declarations, and digests."""
    _require_metadata(
        meta,
        (
            "schema_version",
            "diffgrid_schema_version",
            "opa_type",
            "exojax_version",
            "jax_version",
            "jaxlib_version",
            "created_at",
            "units",
            "array_shapes",
            "array_dtypes",
            "array_digests",
            "opa_state",
        ),
    )
    _validate_schema(str(meta["schema_version"]), allow_downgrade)
    if meta["diffgrid_schema_version"] != DIFFGRID_SCHEMA_VERSION:
        raise ValueError(
            "Saved Diffgrid schema "
            f"{meta['diffgrid_schema_version']} differs from "
            f"{DIFFGRID_SCHEMA_VERSION}."
        )
    if meta["opa_type"] != "OpaDiffgrid":
        raise ValueError(
            "Saved opa_type must be 'OpaDiffgrid'; got "
            f"{meta['opa_type']!r}."
        )
    if strict:
        _ensure_compatible_versions(str(meta["exojax_version"]))

    if meta["units"] != _UNITS:
        raise ValueError("Saved Diffgrid units do not match the supported units.")
    for name in (
        "exojax_version",
        "jax_version",
        "jaxlib_version",
        "created_at",
    ):
        if not isinstance(meta[name], str):
            raise ValueError(f"{name} metadata must be a string.")

    for name in ("aux", "user_meta"):
        if name in meta and not isinstance(meta[name], Mapping):
            raise ValueError(f"{name} metadata must be a mapping.")

    opa_state = meta["opa_state"]
    if not isinstance(opa_state, Mapping):
        raise ValueError("opa_state metadata must be a mapping.")
    if "teacher_method" not in opa_state or "optional_attributes" not in opa_state:
        raise ValueError(
            "opa_state must contain teacher_method and optional_attributes."
        )
    if opa_state["teacher_method"] is not None and not isinstance(
        opa_state["teacher_method"], str
    ):
        raise ValueError("teacher_method metadata must be a string or null.")
    optional_attributes = opa_state["optional_attributes"]
    if not isinstance(optional_attributes, Mapping):
        raise ValueError("optional_attributes metadata must be a mapping.")
    unknown_attributes = set(optional_attributes) - set(
        _OPTIONAL_SCALAR_ATTRIBUTES
    )
    if unknown_attributes:
        raise ValueError(
            "Unsupported optional Diffgrid attributes: "
            f"{', '.join(sorted(unknown_attributes))}."
        )
    if "wavelength_order" in optional_attributes and not isinstance(
        optional_attributes["wavelength_order"], str
    ):
        raise ValueError("wavelength_order metadata must be a string.")
    for name in ("resolution", "molmass"):
        if name in optional_attributes:
            value = optional_attributes[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} metadata must be a finite number.")

    array_shapes = meta["array_shapes"]
    array_dtypes = meta["array_dtypes"]
    array_digests = meta["array_digests"]
    for name, declaration in (
        ("array_shapes", array_shapes),
        ("array_dtypes", array_dtypes),
        ("array_digests", array_digests),
    ):
        if not isinstance(declaration, Mapping):
            raise ValueError(f"{name} metadata must be a mapping.")

    array_names = set(arrays)
    missing_required_arrays = [
        name for name in REQUIRED_DIFFGRID_ARRAYS if name not in arrays
    ]
    if missing_required_arrays:
        raise KeyError(
            "Missing required Diffgrid arrays: "
            f"{', '.join(missing_required_arrays)}"
        )
    supported_array_names = set(REQUIRED_DIFFGRID_ARRAYS) | {"wav"}
    unsupported_array_names = array_names - supported_array_names
    if unsupported_array_names:
        raise ValueError(
            "Unsupported Diffgrid arrays: "
            f"{', '.join(sorted(unsupported_array_names))}."
        )
    for name, declaration in (
        ("array_shapes", array_shapes),
        ("array_dtypes", array_dtypes),
        ("array_digests", array_digests),
    ):
        declared_names = set(declaration)
        if declared_names != array_names:
            missing_arrays = declared_names - array_names
            missing_metadata = array_names - declared_names
            details = []
            if missing_arrays:
                details.append(
                    "arrays missing from the payload: "
                    + ", ".join(sorted(missing_arrays))
                )
            if missing_metadata:
                details.append(
                    "arrays missing metadata: "
                    + ", ".join(sorted(missing_metadata))
                )
            raise ValueError(f"{name} keys do not match payload; {'; '.join(details)}.")

    for name, array in arrays.items():
        if name not in array_shapes:
            raise KeyError(f"Missing shape metadata for Diffgrid array '{name}'.")
        if name not in array_dtypes:
            raise KeyError(f"Missing dtype metadata for Diffgrid array '{name}'.")
        if name not in array_digests:
            raise KeyError(f"Missing digest metadata for Diffgrid array '{name}'.")
        saved_shape_values = array_shapes[name]
        if not isinstance(saved_shape_values, (list, tuple)) or any(
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 0
            for size in saved_shape_values
        ):
            raise ValueError(
                f"Invalid shape metadata for Diffgrid array '{name}'."
            )
        saved_shape = tuple(int(size) for size in saved_shape_values)
        if saved_shape != array.shape:
            raise ValueError(
                f"Shape metadata mismatch for Diffgrid array '{name}': "
                f"expected {saved_shape}, got {array.shape}."
            )
        if array_dtypes[name] != str(array.dtype):
            raise ValueError(
                f"Dtype metadata mismatch for Diffgrid array '{name}': "
                f"expected {array_dtypes[name]!r}, got {str(array.dtype)!r}."
            )

    _validate_array_values(arrays)

    for name, array in arrays.items():
        digest = _sha256_array(array)
        if array_digests[name] != digest:
            raise ValueError(
                f"Digest mismatch for Diffgrid array '{name}'. "
                "Saved file may be corrupted."
            )


def saveopa_diffgrid(
    opa: OpaDiffgrid,
    path: str,
    *,
    format: Literal["zarr", "npz"] = "zarr",
    extra_meta: Mapping[str, Any] | None = None,
    aux: Mapping[str, Any] | None = None,
) -> None:
    """Persist a ready, self-contained ``OpaDiffgrid`` table."""
    from exojax.opacity.diffgrid.api import OpaDiffgrid

    if not isinstance(opa, OpaDiffgrid):
        raise TypeError("saveopa_diffgrid expects an OpaDiffgrid instance.")
    if not getattr(opa, "ready", False):
        raise ValueError("OpaDiffgrid is not ready and cannot be saved.")
    if not hasattr(opa, "diffgrid_info") or opa.diffgrid_info is None:
        raise ValueError("OpaDiffgrid has no Diffgrid table to save.")
    if format not in ("zarr", "npz"):
        raise ValueError("format must be either 'zarr' or 'npz'.")
    path_suffix = Path(path).suffix
    conflicting_suffix = ".npz" if format == "zarr" else ".zarr"
    if path_suffix == conflicting_suffix:
        raise ValueError(
            f"Path suffix '{path_suffix}' conflicts with format='{format}'."
        )

    arrays = _diffgrid_arrays(opa)
    _validate_array_values(arrays)
    meta = _diffgrid_metadata(opa, arrays, extra_meta, aux)
    _validate_archive_metadata(
        arrays,
        meta,
        strict=False,
        allow_downgrade=False,
    )
    if format == "zarr":
        _save_as_zarr(path, arrays, meta)
    else:
        _save_as_npz(path, arrays, meta)


def load_diffgrid_payload(
    path: str,
    *,
    strict: bool = True,
    allow_downgrade: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load and host-validate a self-contained Diffgrid archive."""
    path_object = Path(path)
    if path_object.suffix == ".npz":
        arrays, meta = _load_from_npz(path_object)
    elif path_object.suffix == ".zarr":
        arrays, meta = _load_from_zarr(path_object)
    else:
        arrays, meta = _load(path)
    _validate_archive_metadata(
        arrays,
        meta,
        strict=strict,
        allow_downgrade=allow_downgrade,
    )
    return arrays, meta
