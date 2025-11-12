import os
import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/exojax_numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import exojax.database.multimol as multimol
from exojax.database.multimol import MultiMol, MultiMDBCollection, MultiMDBSnapshot


def test_multimdb_accepts_single_grid_and_emits_snapshot(monkeypatch):
    """Ensure numpy grids are kept intact and snapshot conversion stays offline."""

    captured = {}

    class DummyMdb:
        def __init__(self, path, nu_grid, **kwargs):
            captured["path"] = path
            captured["nu_grid"] = np.array(nu_grid)

        def to_snapshot(self):
            return "dummy-snapshot"

    monkeypatch.setattr(multimol, "MdbExomol", DummyMdb)
    monkeypatch.setattr(
        multimol, "database_path_exomol", lambda mol, root: "CO/12C-16O/SAMPLE"
    )

    mols = [["CO"]]
    dbs = [["ExoMol"]]
    nu_grid = np.linspace(4300.0, 4350.0, 8)

    handler = MultiMol(molmulti=mols, dbmulti=dbs, database_root_path="./")
    multimdb = handler.multimdb(nu_grid, Ttyp=1000.0)

    assert isinstance(multimdb, MultiMDBCollection)
    assert np.array_equal(captured["nu_grid"], nu_grid)

    snap = multimdb.to_snapshot()
    assert isinstance(snap, MultiMDBSnapshot)
    assert snap[0][0] == "dummy-snapshot"
