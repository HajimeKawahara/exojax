"""Compatibility of the legacy nested opacity builder and its new location."""

from types import SimpleNamespace

import numpy as np
import pytest

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.database.multimol import MultiMDBCollection, MultiMol
from exojax.opacity import OpaPremodit
from exojax.opacity.multimol import multiopa_premodit


class SnapshotMDB:
    def __init__(self, grid):
        self.snapshot = MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol", molmass=28.0,
                T_gQT=np.array([300.0, 1000.0, 2000.0]),
                gQT=np.array([1.0, 2.0, 4.0]),
            ),
            lines=Lines(
                nu_lines=grid[[2, len(grid) // 2, -3]],
                elower=np.array([10.0, 20.0, 30.0]),
                line_strength_ref_original=np.array([1.e-23, 2.e-23, 3.e-23]),
            ),
            n_Texp=np.full(3, 0.5),
            alpha_ref=np.full(3, 0.1),
        )

    def to_snapshot(self):
        return self.snapshot


@pytest.fixture
def legacy_inputs():
    grids = [np.geomspace(990.0, 1020.0, 32), np.geomspace(1020.0, 1050.0, 32)]
    mdbs = MultiMDBCollection([
        [SnapshotMDB(grids[0]), SnapshotMDB(grids[0])],
        [SnapshotMDB(grids[1])],
    ])
    handler = MultiMol([["CO", "H2O"], ["H2O"]], [["HITEMP", "HITEMP"], ["HITEMP"]])
    return handler, mdbs, grids


def test_multiopa_single_nu_grid():
    grid = np.geomspace(990.0, 1020.0, 32)
    handler = MultiMol([["CO", "H2O"]], [["HITEMP", "HITEMP"]])
    mdbs = [[SnapshotMDB(grid), SnapshotMDB(grid)]]
    with pytest.warns(DeprecationWarning, match="MultiMol.multiopa_premodit"):
        opas = handler.multiopa_premodit(mdbs, grid, auto_trange=(500.0, 1500.0))
    assert all(isinstance(opa, OpaPremodit) and opa.ready for opa in opas[0])


@pytest.mark.parametrize("snapshot", [False, True])
def test_multiopa_multi_nu_grid(legacy_inputs, snapshot):
    handler, mdbs, grids = legacy_inputs
    payload = mdbs.to_snapshot() if snapshot else mdbs
    with pytest.warns(DeprecationWarning, match="MultiMol.multiopa_premodit"):
        opas = handler.multiopa_premodit(payload, grids, auto_trange=(500.0, 1500.0))
    assert [len(row) for row in opas] == [2, 1]
    for row, grid in zip(opas, grids):
        for opa in row:
            assert isinstance(opa, OpaPremodit)
            np.testing.assert_array_equal(opa.nu_grid, grid)


def test_multiopa_stitching_rejects_indivisible_grid(legacy_inputs):
    handler, mdbs, grids = legacy_inputs
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="cannot be divided"):
        handler.multiopa_premodit(mdbs, grids, (500.0, 1500.0), nstitch_list=[1, 3])


def test_multiopa_preserves_stitching_per_grid(legacy_inputs):
    handler, mdbs, grids = legacy_inputs
    with pytest.warns(DeprecationWarning):
        opas = handler.multiopa_premodit(mdbs, grids, (500.0, 1500.0), nstitch_list=[1, 4])
    assert handler.nstitch_list == [1, 4]
    for row, stitching in zip(opas, [1, 4]):
        assert all(opa.nstitch == stitching for opa in row)


@pytest.mark.parametrize("stitching", [[1], [1, 0], [1, 1.5]])
def test_multiopa_rejects_invalid_segment_stitching(legacy_inputs, stitching):
    _, mdbs, grids = legacy_inputs
    with pytest.raises(ValueError, match="nstitch_list"):
        multiopa_premodit(mdbs, grids, (500.0, 1500.0), nstitch_list=stitching)


def test_opacity_layer_builder_preserves_legacy_values(legacy_inputs):
    handler, mdbs, grids = legacy_inputs
    kwargs = dict(auto_trange=(500.0, 1500.0), nstitch_list=[1, 2])
    direct = multiopa_premodit(mdbs, grids, **kwargs)
    with pytest.warns(DeprecationWarning):
        legacy = handler.multiopa_premodit(mdbs, grids, **kwargs)
    for row, old_row in zip(direct, legacy):
        for opa, old_opa in zip(row, old_row):
            np.testing.assert_array_equal(
                opa.xsmatrix(np.array([1000.0]), np.array([0.1])),
                old_opa.xsmatrix(np.array([1000.0]), np.array([0.1])),
            )


def test_opacity_layer_builder_rejects_segment_count_mismatch(legacy_inputs):
    _, mdbs, grids = legacy_inputs
    with pytest.raises(ValueError, match="same number of segments"):
        multiopa_premodit(mdbs[:1], grids, (500.0, 1500.0))


@pytest.mark.parametrize("nstitch", [1, 2])
def test_legacy_custom_mdb_without_snapshot_and_single_opa_wrapper(nstitch):
    grid = np.geomspace(990.0, 1020.0, 32)
    snapshot = SnapshotMDB(grid).snapshot
    custom_mdb = SimpleNamespace(
        **vars(snapshot.meta), **vars(snapshot.lines),
        n_Texp=snapshot.n_Texp, alpha_ref=snapshot.alpha_ref,
    )
    handler = MultiMol([["CO"]], [["HITEMP"]])
    with pytest.warns(DeprecationWarning, match="store_single_opa"):
        legacy = handler.store_single_opa(
            custom_mdb, grid, (500.0, 1500.0), 0, 0.2, False, nstitch
        )
    direct = OpaPremodit.from_snapshot(
        snapshot, grid, auto_trange=(500.0, 1500.0), nstitch=nstitch
    )
    temperature, pressure = np.array([1000.0]), np.array([0.1])
    np.testing.assert_array_equal(
        legacy.xsmatrix(temperature, pressure), direct.xsmatrix(temperature, pressure)
    )
