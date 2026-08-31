"""Helpers for CKD table precomputation."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from exojax.opacity.ckd.api import OpaCKD


def _patch_ranges(nu_min: float, nu_max: float, patch_width: float):
    left = nu_min
    while left < nu_max:
        right = min(left + patch_width, nu_max)
        if right > left:
            yield left, right
        left = right


def _table_name(prefix: str, ckd_resolution: float, nu_min: float, nu_max: float) -> str:
    return (
        f"{prefix}_R{int(ckd_resolution)}_"
        f"{int(round(nu_min)):05d}_{int(round(nu_max)):05d}.npz"
    )


def _clear_jax_caches() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def precompute_ckd_tables_by_patches(
    make_base_opa: Callable,
    make_nu_grid: Callable,
    nu_min: float,
    nu_max: float,
    patch_width: float,
    T_grid,
    P_grid,
    out_dir,
    *,
    Ng: int = 16,
    ckd_resolution: float = 1000.0,
    band_spacing: str = "log",
    nu_grid_points_per_patch: int = 8000,
    overwrite: bool = False,
    manifest_name: str = "ckd_patch_manifest.json",
    table_prefix: str = "ckd",
) -> dict:
    """Precompute CKD tables by splitting the wavenumber range into patches.

    Args:
        make_base_opa: Callable returning a base opacity calculator for
            ``(nu_grid, patch_min, patch_max)``.
        make_nu_grid: Callable returning a wavenumber grid for
            ``(patch_min, patch_max, n_grid)``.
        nu_min: Lower wavenumber bound in cm-1.
        nu_max: Upper wavenumber bound in cm-1.
        patch_width: Patch width in cm-1.
        T_grid: Temperature grid in K.
        P_grid: Pressure grid in bar.
        out_dir: Directory where patch tables and the manifest are written.
        Ng: Number of g-ordinates.
        ckd_resolution: CKD resolving power used to set each band width.
        band_spacing: Spectral band spacing, either ``"log"`` or ``"linear"``.
        nu_grid_points_per_patch: Number of wavenumber grid points per patch.
        overwrite: Overwrite existing patch tables.
        manifest_name: Manifest filename inside ``out_dir``.
        table_prefix: Prefix for patch table filenames.

    Returns:
        Manifest dictionary describing the generated patch tables.
    """
    if nu_min <= 0.0 or nu_max <= 0.0 or nu_min >= nu_max:
        raise ValueError("nu_min and nu_max must be positive with nu_min < nu_max.")
    if patch_width <= 0.0:
        raise ValueError("patch_width must be positive.")
    if nu_grid_points_per_patch <= 0:
        raise ValueError("nu_grid_points_per_patch must be positive.")
    if band_spacing not in ("log", "linear"):
        raise ValueError("band_spacing must be 'log' or 'linear'.")

    T_grid = np.asarray(T_grid)
    P_grid = np.asarray(P_grid)
    if T_grid.size == 0 or P_grid.size == 0:
        raise ValueError("T_grid and P_grid must not be empty.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name
    ranges = list(_patch_ranges(nu_min, nu_max, patch_width))
    manifest = {
        "schema_version": "ckd_patch_manifest.v1",
        "nu_min_cm-1": float(nu_min),
        "nu_max_cm-1": float(nu_max),
        "patch_width_cm-1": float(patch_width),
        "Ng": int(Ng),
        "ckd_resolution": float(ckd_resolution),
        "band_spacing": band_spacing,
        "nu_grid_points_per_patch": int(nu_grid_points_per_patch),
        "T_grid": T_grid.tolist(),
        "P_grid": P_grid.tolist(),
        "tables": [],
    }

    for index, (patch_min, patch_max) in enumerate(ranges, start=1):
        print(
            f"Patch {index}/{len(ranges)}: "
            f"{patch_min:.3f}-{patch_max:.3f} cm-1, "
            f"nu_grid_points={nu_grid_points_per_patch}"
        )
        nu_grid = make_nu_grid(patch_min, patch_max, nu_grid_points_per_patch)
        base_opa = make_base_opa(nu_grid, patch_min, patch_max)
        band_width = math.sqrt(patch_min * patch_max) / ckd_resolution
        ckd = OpaCKD(base_opa, Ng=Ng, band_width=band_width, band_spacing=band_spacing)
        table_path = out_dir / _table_name(table_prefix, ckd_resolution, patch_min, patch_max)
        ckd.precompute_tables(T_grid, P_grid, to_path=str(table_path), overwrite=overwrite)

        manifest["tables"].append(
            {
                "index": index,
                "nu_min_cm-1": float(patch_min),
                "nu_max_cm-1": float(patch_max),
                "path": str(table_path),
                "n_bands": int(np.asarray(ckd.nu_bands).size),
                "band_width_cm-1": float(band_width),
            }
        )
        # Write after each patch so long runs leave usable progress.
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # Drop references before the next line-list patch is opened.
        del ckd
        del base_opa
        del nu_grid
        gc.collect()
        _clear_jax_caches()

    return manifest
