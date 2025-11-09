from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from exojax import __version__ as _EXOJAX_VERSION

SCHEMA_OPA_VERSION = "ioopa@1"


def _sha256_array(arr: np.ndarray) -> str:
    """Return SHA-256 digest of the array contents, dtype, and shape."""
    m = hashlib.sha256()
    m.update(arr.tobytes(order="C"))
    m.update(str(arr.dtype).encode())
    m.update(str(arr.shape).encode())
    return m.hexdigest()


def _digest_snapshot_meta(snapshot_meta: Dict[str, Any]) -> str:
    """Compute a digest for snapshot metadata."""
    payload = json.dumps(snapshot_meta, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(arrays: Dict[str, np.ndarray], keys: Iterable[str]) -> None:
    """Ensure expected array keys exist."""
    missing = [k for k in keys if k not in arrays]
    if missing:
        raise KeyError(f"Missing required opa arrays: {', '.join(missing)}")


def _ensure_compatible_versions(saved_version: str) -> None:
    """Guard against loading data produced by a different ExoJAX version."""
    if saved_version != _EXOJAX_VERSION:
        raise ValueError(
            "Saved opa was created with ExoJAX "
            f"{saved_version}, but current version is {_EXOJAX_VERSION}. "
            "Pass strict=False to allow loading anyway (unsupported)."
        )


def _validate_schema(schema_version: str, allow_downgrade: bool) -> None:
    """Verify the serialization schema matches the current reader."""
    if schema_version == SCHEMA_OPA_VERSION:
        return
    if not allow_downgrade:
        raise ValueError(
            f"Saved opa schema {schema_version} differs from "
            f"{SCHEMA_OPA_VERSION}. Set allow_downgrade=True to continue."
        )


def _load(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load serialized opa arrays + metadata from NPZ or Zarr."""
    path_obj = Path(path)
    if path_obj.suffix == ".npz" or path_obj.with_suffix(".npz").exists():
        npz_path = path_obj if path_obj.suffix == ".npz" else path_obj.with_suffix(
            ".npz"
        )
        return _load_from_npz(npz_path)
    if (
        path_obj.suffix == ".zarr"
        or path_obj.with_suffix(".zarr").exists()
        or path_obj.is_dir()
    ):
        zarr_path = (
            path_obj
            if (path_obj.suffix == ".zarr" or path_obj.is_dir())
            else path_obj.with_suffix(".zarr")
        )
        return _load_from_zarr(zarr_path)
    raise FileNotFoundError(
        f"Cannot resolve opa file at '{path}'. Expected .npz or .zarr artifact."
    )


def _load_from_npz(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arrays = dict(np.load(npz_path, allow_pickle=False))
    meta_path = npz_path.with_suffix("").with_name(npz_path.stem + "_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{meta_path}' missing for opa archive '{npz_path}'."
        )
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    return arrays, meta


def _load_from_zarr(zarr_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    import zarr  # lazy import so users without zarr can still load .npz files

    group = zarr.open(zarr_path, mode="r")
    arrays = {name: np.asarray(group[name]) for name in group.keys()}
    meta = dict(group.attrs)
    group.close()
    return arrays, meta
