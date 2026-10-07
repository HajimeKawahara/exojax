"""Legacy multi-database loading without production mock dependencies."""

from pathlib import Path

import numpy as np
import pytest

import exojax.database.multimol as multimol
from exojax.database.multimol import MultiMol


@pytest.fixture
def loader(monkeypatch):
    class DummyMDB:
        def __init__(self, path, nu_grid, **kwargs):
            self.path = path
            self.nu_grid = nu_grid
            self.options = kwargs

    for name in ("MdbExomol", "MdbHitran", "MdbHitemp"):
        monkeypatch.setattr(multimol, name, DummyMDB)
    monkeypatch.setattr(multimol, "database_path_exomol", lambda mol, root: mol)
    monkeypatch.setattr(multimol, "database_path_hitran12", lambda mol: mol)
    return DummyMDB


@pytest.mark.parametrize("database", ["ExoMol", "HITRAN12", "HITEMP"])
def test_multimdb_single_nu_grid(loader, database):
    handler = MultiMol([["CO", "H2O"]], [[database, database]])
    grid = np.geomspace(990.0, 1020.0, 32)
    mdbs = handler.multimdb([grid], crit=1.e-30, Ttyp=1200.0)

    assert len(mdbs) == 1
    assert len(mdbs[0]) == 2
    for mdb in mdbs[0]:
        assert isinstance(mdb, loader)
        np.testing.assert_array_equal(mdb.nu_grid, grid)
        assert mdb.options["gpu_transfer"] is False
        assert mdb.options["crit"] == 1.e-30
        assert mdb.options["Ttyp"] == 1200.0
    assert handler.mols_unique == ["CO", "H2O"]


def test_multimol_different_structure_raise_error():
    with pytest.raises(ValueError, match="different structures"):
        MultiMol([["CO", "H2O"], ["H2O"]], [["HITEMP", "HITEMP"]])


def test_no_lines_only_masks_legacy_result(loader, monkeypatch):
    def load(path, nu_grid, **kwargs):
        if Path(path).name == "CO":
            raise ValueError("No line found in ", [990.0, 1020.0], "cm-1")
        return loader(path, nu_grid, **kwargs)

    monkeypatch.setattr(multimol, "MdbExomol", load)
    names = [["CO", "H2O"]]
    handler = MultiMol(names, [["ExoMol", "ExoMol"]])
    with pytest.warns(UserWarning, match="CO.*no selected lines"):
        result = handler.multimdb(np.geomspace(990.0, 1020.0, 32))

    assert names == [["CO", "H2O"]]
    assert handler.molmulti == names
    assert handler.masked_molmulti == [["H2O"]]
    assert handler.mols_num == [[0]]
    assert len(result[0]) == 1


@pytest.mark.parametrize("error", [OSError("download failed"), ValueError("invalid data")])
def test_load_failures_propagate_instead_of_exiting(loader, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(multimol, "MdbExomol", fail)
    handler = MultiMol([["CO"]], [["ExoMol"]])
    with pytest.raises(type(error), match=str(error)):
        handler.multimdb(np.geomspace(990.0, 1020.0, 32))


def test_sample_is_not_a_production_backend():
    with pytest.raises(ValueError, match="Unsupported database: SAMPLE"):
        MultiMol([["CO"]], [["SAMPLE"]])
