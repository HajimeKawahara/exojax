"""Unit tests for CKD patch precomputation helpers."""

import json
import os

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from exojax.opacity import OpaCKD
from exojax.opacity.ckd.precompute import precompute_ckd_tables_by_patches


class MockBaseOpa:
    """Small base opacity calculator for patch precompute tests."""

    def __init__(self, nu_grid):
        self.nu_grid = np.asarray(nu_grid)

    def xsmatrix(self, T_grid, P_grid):
        T_grid = np.asarray(T_grid)
        P_grid = np.asarray(P_grid)
        nu_scale = self.nu_grid / self.nu_grid[0]
        return (
            1.0e-24
            * T_grid[:, None]
            / T_grid[0]
            * P_grid[:, None]
            / P_grid[0]
            * nu_scale[None, :]
        )


def test_precompute_ckd_tables_by_patches_writes_manifest_and_tables(tmp_path):
    def make_nu_grid(nu_min, nu_max, n_grid):
        return np.linspace(nu_min, nu_max, n_grid)

    def make_base_opa(nu_grid, _nu_min, _nu_max):
        return MockBaseOpa(nu_grid)

    manifest = precompute_ckd_tables_by_patches(
        make_base_opa,
        make_nu_grid,
        1000.0,
        1020.0,
        10.0,
        np.array([500.0, 1000.0]),
        np.array([1.0e-3, 1.0e-1]),
        tmp_path,
        Ng=4,
        ckd_resolution=100.0,
        nu_grid_points_per_patch=32,
        overwrite=True,
        table_prefix="test_ckd",
    )

    manifest_path = tmp_path / "ckd_patch_manifest.json"
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest == manifest
    assert loaded_manifest["schema_version"] == "ckd_patch_manifest.v1"
    assert len(loaded_manifest["tables"]) == 2

    for table in loaded_manifest["tables"]:
        path = tmp_path / table["path"]
        assert path.exists()
        ckd = OpaCKD.load_only().load_tables(str(path))
        assert ckd.ready
        assert ckd.ckd_info.log_kggrid.shape[:3] == (2, 2, 4)
        assert ckd.ckd_info.log_kggrid.shape[3] == table["n_bands"]


def test_precompute_ckd_tables_by_patches_validates_inputs(tmp_path):
    def make_nu_grid(_nu_min, _nu_max, _n_grid):
        return np.linspace(1000.0, 1010.0, 16)

    def make_base_opa(nu_grid, _nu_min, _nu_max):
        return MockBaseOpa(nu_grid)

    with pytest.raises(ValueError, match="patch_width"):
        precompute_ckd_tables_by_patches(
            make_base_opa,
            make_nu_grid,
            1000.0,
            1010.0,
            0.0,
            np.array([500.0]),
            np.array([1.0e-3]),
            tmp_path,
        )
