import os
import json
import hashlib
import sys
import platform
from datetime import datetime, timezone
import numpy as np
from jax import Array
from importlib.metadata import version as _pkg_version

from exojax.opacity.io.serialization import _sha256_array

def _safe_version(pkg: str) -> str:
    try: return _pkg_version(pkg)
    except Exception: return "unknown"

def _hash_json(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

def _fingerprint_value(value):
    """Represent numerical state without embedding large arrays in metadata."""
    if isinstance(value, (np.ndarray, Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot identify opacity data")
        return dict(sha256=_sha256_array(array))
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _fingerprint_value(item) for key, item in value.items()}
    raise TypeError("Non-data attribute")

def _state_fingerprint(obj) -> dict:
    """Capture direct data attributes, excluding opaque runtime objects."""
    state = {}
    for name, value in vars(obj).items():
        if name in ("warning", "ready"):
            continue
        try:
            state[name] = _fingerprint_value(value)
        except TypeError:
            # Do not traverse functions, dataframe internals or runtime wrappers.
            # Custom calculators can describe additional inputs in meta().
            continue
    return dict(
        class_name=f"{type(obj).__module__}.{type(obj).__qualname__}",
        state=state,
    )

def _base_fingerprint(base_opa) -> dict:
    """Identify full grids, line data, partition functions and calculation settings."""
    fingerprint = _state_fingerprint(base_opa)
    fingerprint["fingerprint_version"] = 2
    fingerprint["nu_grid"] = _fingerprint_value(base_opa.nu_grid)
    fingerprint["base_meta"] = _fingerprint_value(
        getattr(base_opa, "meta", lambda: {})()
    )
    for name in (
        "mdb", "pf_provider", "broadening_strategy", "pre_modit_info", "diffgrid_info"
    ):
        component = getattr(base_opa, name, None)
        if component is not None:
            fingerprint[name] = _state_fingerprint(component)
    return fingerprint

def _ckd_metadata_dict(self) -> dict:
    if self.base_opa is None:
        base_fp = getattr(self, "_expected_base_meta", None)
        if base_fp is None:
            raise ValueError("Cannot save CKD table without base fingerprint metadata.")
    else:
        base_fp = _base_fingerprint(self.base_opa)
    return dict(
        schema_version="ckd.v2",
        exojax_version=_safe_version("ExoJAX"),
        jax_version=_safe_version("jax"),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        created_at=datetime.now(timezone.utc).isoformat(),
        base_fingerprint=base_fp,
        base_fingerprint_hash=_hash_json(base_fp),
        Ng=int(self.Ng),
        band_width=float(self.band_width),
        band_spacing=str(self.band_spacing),
        n_bands=int(len(self.nu_bands)),
        units=dict(P="bar", nu="cm^-1", k="cm^2"),
        shapes=dict(log_kggrid=list(map(int, self.ckd_info.log_kggrid.shape))),
        dtypes=dict(log_kggrid=str(self.ckd_info.log_kggrid.dtype)),
    )

def _ckd_save_as_npz(self, path: str, overwrite: bool=False) -> None:
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"{path} already exists")
    info = self.ckd_info
    meta_json = json.dumps(_ckd_metadata_dict(self)).encode("utf-8")
    np.savez_compressed(
        path,
        log_kggrid=np.asarray(info.log_kggrid),
        ggrid=np.asarray(info.ggrid),
        weights=np.asarray(info.weights),
        T_grid=np.asarray(info.T_grid),
        P_grid=np.asarray(info.P_grid),
        nu_bands=np.asarray(info.nu_bands),
        band_edges=np.asarray(info.band_edges),
        meta=np.frombuffer(meta_json, dtype=np.uint8),
    )
