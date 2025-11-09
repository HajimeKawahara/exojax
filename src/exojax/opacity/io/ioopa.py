from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal

import jax
import jaxlib
import numpy as np

from exojax import __version__
from exojax.opacity.base import OpaCalc
from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.io.serialization import SCHEMA_OPA_VERSION
from exojax.opacity.io.serialization import _sha256_array
from exojax.opacity.lpf.api import OpaDirect
from exojax.opacity.modit.api import OpaModit
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.providers import (
    ExomolPartitionProvider,
    HitranPartitionProvider,
    ExomolBroadening,
    HitranBroadening,
)


def saveopa(
    opa: OpaCalc,
    path: str,
    *,
    format: Literal["zarr", "npz"] = "zarr",
    extra_meta: Dict[str, Any] | None = None,
) -> None:
    """Generic entry point for persisting ``Opa*`` calculators to disk.

    Currently only :class:`OpaPremodit` is implemented; other calculators raise
    ``NotImplementedError`` placeholders (``saveopa_ckd``, ``saveopa_modit``,
    ``saveopa_direct``) to document the expected extension points.
    """
    if isinstance(opa, OpaPremodit):
        saveopa_premodit(opa, path, format=format, extra_meta=extra_meta)
        return
    if isinstance(opa, OpaCKD):
        raise NotImplementedError(
            "saveopa_ckd is not implemented yet for OpaCKD instances."
        )
    if isinstance(opa, OpaModit):
        raise NotImplementedError(
            "saveopa_modit is not implemented yet for OpaModit instances."
        )
    if isinstance(opa, OpaDirect):
        raise NotImplementedError(
            "saveopa_direct is not implemented yet for OpaDirect instances."
        )
    raise TypeError(
        "saveopa does not support persisting instances of "
        f"{opa.__class__.__name__}."
    )


def saveopa_premodit(
    opa: OpaPremodit,
    path: str,
    *,
    format: Literal["zarr", "npz"] = "zarr",
    extra_meta: Dict[str, Any] | None = None,
) -> None:
    """Persist an initialized ``OpaPremodit`` to disk."""
    if not isinstance(opa, OpaPremodit):
        raise TypeError("saveopa_premodit expects an OpaPremodit instance.")
    if not getattr(opa, "ready", False):
        raise ValueError("OpaPremodit is not ready. Call apply_params() before saving.")
    for attr in ("gamma_ref", "n_Texp", "ngrid_broadpar", "ngrid_elower"):
        if not hasattr(opa, attr):
            raise ValueError(f"OpaPremodit missing attribute '{attr}' required for saving.")

    info_tuple = opa._get_info_tuple()
    (
        multi_index_uniqgrid,
        elower_grid,
        ngamma_ref_grid,
        n_Texp_grid,
        R,
        pmarray,
    ) = info_tuple

    arrays: Dict[str, np.ndarray] = {
        "nu_grid": np.asarray(opa.nu_grid),
        "multi_index_uniqgrid": np.asarray(multi_index_uniqgrid),
        "elower_grid": np.asarray(elower_grid),
        "ngamma_ref_grid": np.asarray(ngamma_ref_grid),
        "n_Texp_grid": np.asarray(n_Texp_grid),
        "R": np.asarray(R),
        "pmarray": np.asarray(pmarray),
        "gamma_ref": np.asarray(opa.gamma_ref),
        "n_Texp": np.asarray(opa.n_Texp),
    }

    lbd_layout = None
    if hasattr(opa, "lbd_coeff_reshaped"):
        arrays["lbd_coeff_reshaped"] = np.asarray(opa.lbd_coeff_reshaped)
        lbd_layout = "reshaped"
    elif hasattr(opa, "lbd_coeff"):
        arrays["lbd_coeff"] = np.asarray(opa.lbd_coeff)
        lbd_layout = "flat"
    else:
        raise ValueError("OpaPremodit instance lacks lbd_coeff data; cannot save.")

    pf_meta = _serialize_pf_provider(opa, arrays)
    broadening_meta = _serialize_broadening_strategy(opa, arrays)

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_OPA_VERSION,
        "exojax_version": __version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "opa_type": "OpaPremodit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": str(arrays["nu_grid"].dtype),
        "nu_grid_digest": _sha256_array(arrays["nu_grid"]),
        "opa_state": {
            "dbtype": opa.dbtype,
            "molmass": float(opa.molmass),
            "diffmode": int(opa.diffmode),
            "wavelength_order": opa.wavelength_order,
            "version_auto_trange": int(opa.version_auto_trange),
            "dit_grid_resolution": (
                float(opa.dit_grid_resolution)
                if opa.dit_grid_resolution is not None
                else None
            ),
            "single_broadening": bool(opa.single_broadening),
            "single_broadening_parameters": (
                list(opa.single_broadening_parameters)
                if opa.single_broadening_parameters is not None
                else None
            ),
            "dE": float(opa.dE),
            "Tref": float(opa.Tref),
            "Twt": float(opa.Twt),
            "Tmax": float(opa.Tmax),
            "Tmin": float(opa.Tmin),
            "Tref_broadening": float(opa.Tref_broadening),
            "method": opa.method,
            "warning": bool(getattr(opa, "warning", False)),
            "cutwing": float(opa.cutwing),
            "nstitch": int(opa.nstitch),
            "alias": opa.alias,
            "ngrid_broadpar": int(opa.ngrid_broadpar),
            "ngrid_elower": int(opa.ngrid_elower),
        },
        "lbd_layout": lbd_layout,
        "provider_contracts": {
            "partition": pf_meta,
            "broadening": broadening_meta,
        },
    }

    if extra_meta:
        meta["user_meta"] = extra_meta

    if format == "zarr":
        _save_as_zarr(path, arrays, meta)
    else:
        _save_as_npz(path, arrays, meta)

def _save_as_npz(path: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    import json
    import numpy as np
    import os

    # Save arrays as .npz
    npz_path = path if path.endswith(".npz") else path + ".npz"
    np.savez_compressed(npz_path, **arrays)

    # Save metadata as JSON
    meta_path = os.path.splitext(npz_path)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)    

def _save_as_zarr(path: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    import zarr

    # Create a Zarr group
    zarr_path = path if path.endswith(".zarr") else path + ".zarr"
    zarr_group = zarr.open(zarr_path, mode="w")

    # Save arrays
    for name, array in arrays.items():
        zarr_group.create_dataset(name, data=array, compressor=zarr.get_codec({'id': 'zlib', 'level': 1}))

    # Save metadata
    zarr_group.attrs.update(meta)
    # zarr.Group lacks close() on some versions; close what we can.
    close_group = getattr(zarr_group, "close", None)
    if callable(close_group):
        close_group()
    else:
        store = getattr(zarr_group, "store", None)
        close_store = getattr(store, "close", None)
        if callable(close_store):
            close_store()


def _serialize_pf_provider(opa: OpaPremodit, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Capture the partition-function provider configuration."""
    provider = getattr(opa, "pf_provider", None)
    if provider is None:
        raise ValueError("OpaPremodit has no partition-function provider attached.")

    if isinstance(provider, ExomolPartitionProvider):
        arrays["pf_T_gQT"] = np.asarray(provider.T_gQT)
        arrays["pf_gQT"] = np.asarray(provider.gQT)
        return {"kind": "exomol"}
    if isinstance(provider, HitranPartitionProvider):
        arrays["pf_T_gQT"] = np.asarray(provider.T_gQT)
        arrays["pf_gQT"] = np.asarray(provider.gQT)
        if provider.uniqiso is not None:
            arrays["pf_uniqiso"] = np.asarray(provider.uniqiso)
        return {"kind": "hitran", "isotope": int(provider.isotope)}
    raise TypeError(
        "Unsupported partition-function provider for serialization: "
        f"{provider.__class__.__name__}"
    )


def _serialize_broadening_strategy(opa: OpaPremodit, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Capture the broadening strategy configuration."""
    strategy = getattr(opa, "broadening_strategy", None)
    if strategy is None:
        raise ValueError("OpaPremodit has no broadening strategy attached.")

    if isinstance(strategy, ExomolBroadening):
        arrays["broadening_n_Texp_template"] = np.asarray(strategy._n_Texp)
        arrays["broadening_alpha_ref"] = np.asarray(strategy._alpha_ref)
        return {"kind": "exomol"}
    if isinstance(strategy, HitranBroadening):
        arrays["broadening_n_air"] = np.asarray(strategy._n_air)
        arrays["broadening_gamma_air"] = np.asarray(strategy._gamma_air)
        return {"kind": "hitran"}
    raise TypeError(
        "Unsupported broadening strategy for serialization: "
        f"{strategy.__class__.__name__}"
    )
