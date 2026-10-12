"""Supplemental preparation shares grids while pinning its own provenance."""

import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_supplemental")


def arguments(preparation, tmp_path, extra=()):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema_version": preparation.base.SCHEMA_VERSION,
        "column": {"pressure_boundaries_bar": [.001, .01, .1]}}))
    return ["--database-root", str(tmp_path / "database"), "--inventory", str(inventory),
        "--species", "H1O1", "--nu-min", "4329", "--nu-max", "4335",
        "--resolved-nu-min", "4329", "--resolved-nu-max", "4335",
        "--wing-padding", "1", "--samples-per-band", "64",
        "--ckd-resolution", "1000", "--bands-per-chunk", "2",
        "--temperature-nodes", "2", "--ng", "2", *extra]


def test_dry_run_preserves_base_grid_and_no_writes(preparation, tmp_path, capsys):
    argv = arguments(preparation, tmp_path)
    original_sources = dict(preparation.base.LINE_LISTS)
    base_hash = preparation.base._sha256(Path(preparation.base.__file__))
    args = preparation.parser_with_defaults().parse_args(argv)
    plan = preparation.make_plan(args)
    assert plan["mdb_path"].endswith("OH/16O-1H/MYTHOS")
    assert plan["output_dir"].endswith("prepared/supplemental")
    args.mdb_path = Path(plan["mdb_path"])
    baseline = preparation.base.make_plan(args)
    assert plan["grid_fingerprint"] == baseline["grid_fingerprint"]
    assert plan["settings"] == baseline["settings"]
    assert "supplemental_preparation" not in plan["settings"]
    assert preparation.main(argv + ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["species"] == "H1O1"
    assert not Path(plan["output_dir"]).exists()
    assert preparation.base.LINE_LISTS == original_sources
    assert preparation.base._sha256(Path(preparation.base.__file__)) == base_hash


def test_species_match_fastchem_setup(preparation):
    standard = pytest.importorskip("exogibbs.thermo.standard",
                                  reason="Optional RCE/TCE chemistry dependency")
    species = standard.prepare_fastchem_thermodynamics().chemical_setup.species
    assert set(preparation.SUPPLEMENTAL_LINE_LISTS) <= set(species)


def test_bundled_co_alias_child_dispatch_and_resume(preparation, tmp_path, monkeypatch):
    """Use tiny CO input under an OH dispatch label, solely as an offline fixture."""
    from exojax.database.exomol.api import MdbExomol
    from exojax.test.data import get_testdata_filename

    database = tmp_path / "CO"
    shutil.copytree(Path(get_testdata_filename("CO")), database)
    source = database / "12C-16O" / "SAMPLE"

    def offline_database(*args, **kwargs):
        kwargs["broadf_download"] = False
        return MdbExomol(*args, **kwargs)

    monkeypatch.setattr("exojax.database.exomol.api.MdbExomol", offline_database)
    argv = arguments(preparation, tmp_path, ["--mdb-path", str(source)])
    calls = []

    def child(command, check):
        assert check
        assert Path(command[1]).resolve() == Path(preparation.__file__).resolve()
        assert command[-2:] == ["--chunk-index", "0"]
        calls.append(command)
        assert preparation.main(command[2:]) == 0

    monkeypatch.setattr(preparation, "subprocess", SimpleNamespace(run=child))
    # main changes these only for host merging; restore the test environment.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    assert preparation.main(argv) == 0
    assert len(calls) == 1
    plan = preparation.make_plan(preparation.parser_with_defaults().parse_args(argv))
    records = preparation.completed_chunks(plan)
    assert records[0]["provenance"]["number_of_lines"] > 0
    assert records[0]["provenance"]["supplemental_preparation"] == plan["supplemental_preparation"]
    complete_path = Path(plan["output_dir"]) / "H1O1" / "complete.json"
    complete = json.loads(complete_path.read_text())
    assert complete["plan"]["grid_fingerprint"] == plan["grid_fingerprint"]
    metadata = json.loads(Path(complete["files"]["diffgrid_metadata.json"]["path"]).read_text())
    assert plan["supplemental_preparation"]["driver_sha256"] in json.dumps(metadata)
    with np.load(complete["files"]["ckd.npz"]["path"]) as tables:
        assert not np.isnan(tables["log_kggrid"]).any()
    assert preparation.main(argv) == 0
    assert len(calls) == 1
    changed = dict(plan, supplemental_preparation=dict(plan["supplemental_preparation"], driver_sha256="changed"))
    with pytest.raises(ValueError, match="different driver"):
        preparation.completed_chunks(changed)
    altered_grid = dict(plan, grid_fingerprint="changed")
    with pytest.raises(ValueError, match="different settings"):
        preparation.completed_chunks(altered_grid)
