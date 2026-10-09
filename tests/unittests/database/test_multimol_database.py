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


@pytest.mark.parametrize("database", ["ExoMol", "exomol", "HITRAN12", "hitran12", "HITEMP", "hitemp"])
def test_multimdb_single_nu_grid(loader, database):
    handler = MultiMol(
        [["CO", "H2O"]], [[database, database]], database_root_path="test-databases"
    )
    grid = np.geomspace(990.0, 1020.0, 32)
    mdbs = handler.multimdb([grid], crit=1.e-30, Ttyp=1200.0)

    assert len(mdbs) == 1
    assert len(mdbs[0]) == 2
    provider_options = {"broadf_download": False} if database.lower() == "exomol" else {"isotope": 1}
    for mdb, directory in zip(mdbs[0], handler.db_dirs[0]):
        assert isinstance(mdb, loader)
        assert Path(mdb.path) == Path("test-databases") / directory
        np.testing.assert_array_equal(mdb.nu_grid, grid)
        assert mdb.options == {
            "gpu_transfer": False, "crit": 1.e-30, "Ttyp": 1200.0,
            **provider_options,
        }
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


def test_failed_reload_preserves_last_complete_selection(loader, monkeypatch):
    grids = [np.geomspace(990.0, 1020.0, 32), np.geomspace(1030.0, 1060.0, 32)]
    state = {"empty_first_co": True, "fail_second_segment": False}

    def load(path, nu_grid, **kwargs):
        if nu_grid is grids[1] and state["fail_second_segment"]:
            raise OSError("Second segment download failed.")
        if nu_grid is grids[0] and Path(path).name == "CO" and state["empty_first_co"]:
            raise ValueError("No line found in ", [990.0, 1020.0], "cm-1")
        return loader(path, nu_grid, **kwargs)

    monkeypatch.setattr(multimol, "MdbExomol", load)
    handler = MultiMol(
        [["CO", "H2O"], ["CO"]], [["ExoMol", "ExoMol"], ["ExoMol"]]
    )
    with pytest.warns(UserWarning, match="CO.*no selected lines"):
        handler.multimdb(grids)
    previous = (handler.masked_molmulti, handler.mols_unique, handler.mols_num)
    assert previous == ([["H2O"], ["CO"]], ["H2O", "CO"], [[0], [1]])

    state.update(empty_first_co=False, fail_second_segment=True)
    with pytest.raises(OSError, match="Second segment"):
        handler.multimdb(grids)
    assert handler.masked_molmulti is previous[0]
    assert handler.mols_unique is previous[1]
    assert handler.mols_num is previous[2]

    state["fail_second_segment"] = False
    handler.multimdb(grids)
    assert handler.masked_molmulti == [["CO", "H2O"], ["CO"]]
    assert handler.mols_unique == ["CO", "H2O"]
    assert handler.mols_num == [[0, 1], [0]]


def test_unique_indices_preserve_first_occurrence_and_empty_segments(loader):
    names = [["CO", "H2O", "CO"], [], ["H2O", "CH4", "CO"]]
    handler = MultiMol(names, [["ExoMol"] * len(row) for row in names])
    grid = np.geomspace(990.0, 1020.0, 32)

    handler.multimdb([grid] * len(names))

    assert handler.mols_unique == ["CO", "H2O", "CH4"]
    assert handler.mols_num == [[0, 1, 0], [], [1, 2, 0]]


def test_caller_mutation_does_not_change_configured_database_paths(loader):
    names = [["CO", "H2O"]]
    databases = [["ExoMol", "ExoMol"]]
    handler = MultiMol(names, databases)
    names[0][0] = "CH4"
    names.append(["CO"])
    databases[0][0] = "HITEMP"
    databases.append(["HITEMP"])

    result = handler.multimdb(np.geomspace(990.0, 1020.0, 32))

    assert handler.molmulti == [["CO", "H2O"]]
    assert handler.dbmulti == [["ExoMol", "ExoMol"]]
    assert [Path(mdb.path).name for mdb in result[0]] == ["CO", "H2O"]
    assert all(mdb.options["broadf_download"] is False for mdb in result[0])


def test_sample_is_not_a_production_backend():
    with pytest.raises(ValueError, match="Unsupported database: SAMPLE"):
        MultiMol([["CO"]], [["SAMPLE"]])
