import os
import json
import hashlib
import sys
import platform
import datetime
import numpy as np
from importlib.metadata import version as _pkg_version

def _safe_version(pkg: str) -> str:
    try: return _pkg_version(pkg)
    except Exception: return "unknown"

def _hash_json(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

def _base_fingerprint(base_opa) -> dict:
    return dict(
        class_name=type(base_opa).__name__,
        nu_min=float(base_opa.nu_grid[0]),
        nu_max=float(base_opa.nu_grid[-1]),
        nu_len=int(len(base_opa.nu_grid)),
        base_meta=getattr(base_opa, "meta", lambda: {})(),
    )

def _ckd_metadata_dict(self) -> dict:
    base_fp = _base_fingerprint(self.base_opa)
    return dict(
        schema_version="ckd.v1",
        exojax_version=_safe_version("ExoJAX"),
        jax_version=_safe_version("jax"),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        created_at=datetime.datetime.utcnow().isoformat()+"Z",
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

