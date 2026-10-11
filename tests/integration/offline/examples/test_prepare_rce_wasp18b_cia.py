"""Verify physical CIA unit conversion and the existing database interface."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_cia")


def test_borysow_units_and_cdb_roundtrip(preparation, tmp_path):
    from jax.experimental import enable_x64
    from exojax.database.cia.api import CdbCIA

    source = tmp_path / "CIA.H2H2.Yi"
    source.write_text("===\nfreq(cm-1) 1000K 4000K 7000K\n===\n"
                      "20 1.0e-5 2.0e-5 3.0e-5\n40 4.0e-5 5.0e-5 6.0e-5\n")
    converted = tmp_path / "H2-H2_Borysow2001.cia"
    metadata = preparation.convert_borysow(source, converted)
    with enable_x64():
        cdb = CdbCIA(str(converted))
        actual = np.asarray(10.0 ** cdb.logac)
    np.testing.assert_array_equal(cdb.nucia, [20.0, 40.0])
    np.testing.assert_array_equal(cdb.tcia, [1000.0, 4000.0, 7000.0])
    expected = np.asarray([[1., 4.], [2., 5.], [3., 6.]]) * 1e-5 / (2.68676e19)**2
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert metadata["temperature_range_K"] == [1000.0, 7000.0]


def test_h2he_temperature_domain_is_required(preparation, tmp_path):
    source = tmp_path / "H2-He_2011.cia"
    source.write_text("H2-He 20 40 2 1000\n20 1e-45\n40 2e-45\n"
                      "H2-He 20 40 2 3000\n20 2e-45\n40 3e-45\n")
    with pytest.raises(ValueError, match="covering 1500--4500"):
        preparation.validate_hitran_h2he(source)


@pytest.mark.parametrize("header,rows,message", [
    ("20 40", "20 1e-45\n31 2e-45\n40 3e-45\n", "same wavenumber grid"),
    ("20 40", "20 1e-45\n40 2e-45\n30 3e-45\n", "increasing positive"),
    ("20 40", "20 1e-45\n20 2e-45\n40 3e-45\n", "increasing positive"),
    ("20 40", "0 1e-45\n30 2e-45\n40 3e-45\n", "increasing positive"),
    ("20 40", "20 1e-45\nnan 2e-45\n40 3e-45\n", "increasing positive"),
    ("20 40", "20 1e-45\n30 nan\n40 3e-45\n", "finite nonnegative"),
    ("20 40", "20 1e-45\n30 -2e-45\n40 3e-45\n", "finite nonnegative"),
    ("10 40", "20 1e-45\n30 2e-45\n40 3e-45\n", "endpoints"),
])
def test_h2he_checks_every_temperature_block(preparation, tmp_path, header, rows, message):
    source = tmp_path / "H2-He_2011.cia"
    source.write_text("H2-He 20 40 3 1000\n20 1e-45\n30 2e-45\n40 3e-45\n"
                      f"H2-He {header} 3 7000\n{rows}")
    with pytest.raises(ValueError, match=message):
        preparation.validate_hitran_h2he(source)


def test_dry_run_leaves_output_absent(preparation, tmp_path):
    preparation.main(["--database-root", str(tmp_path), "--dry-run"])
    assert not (tmp_path / "rce_wasp18b").exists()


def test_original_inventory_cannot_be_updated(preparation, tmp_path):
    with pytest.raises(SystemExit):
        preparation.main(["--manifest", str(tmp_path / "local_inventory.json")])


@pytest.mark.parametrize("h2_key,he_key,missing_helium", [
    ("H2-H2", "H2-He", False),
    ("cia_hydrogen", "cia_helium", False),
    ("cia_hydrogen", "cia_helium", True),
])
def test_preparation_updates_only_working_manifest(
        preparation, tmp_path, h2_key, he_key, missing_helium):
    from _rce_wasp18b_assets import load_manifest

    root = tmp_path / "rce_wasp18b"
    output = root / "prepared" / "high_temperature"
    output.mkdir(parents=True)
    (output / "CIA.H2H2.Yi").write_text(
        "===\nfreq(cm-1) 1000K 4000K 7000K\n===\n"
        "20 1e-5 2e-5 3e-5\n40 4e-5 5e-5 6e-5\n")
    helium = (output if missing_helium else root) / "H2-He_2011.cia"
    helium.write_text("H2-He 20 40 2 1000\n20 1e-45\n40 2e-45\n"
                      "H2-He 20 40 2 7000\n20 2e-45\n40 3e-45\n")
    config = {
        "schema_version": preparation.SCHEMA_VERSION,
        "files": {
            h2_key: {"path": str(root / "missing-H2-H2.cia"), "sha256": "0" * 64,
                     "source": "unavailable old table"},
            he_key: {"path": str(root / "missing-H2-He.cia") if missing_helium else str(helium),
                     "sha256": preparation._sha256(helium), "source": "offline fixture"},
        },
        "cia": {"H2-H2": {"asset": h2_key}, "H2-He": {"asset": he_key}},
    }
    inventory = root / "local_inventory.json"
    inventory.write_text(json.dumps(config))
    working = output / "manifest.json"
    working.write_bytes(inventory.read_bytes())
    preparation.main(["--database-root", str(tmp_path), "--manifest", str(working)])
    manifest = load_manifest(working)
    assert json.loads(inventory.read_text()) == config
    assert manifest.paths[he_key] == helium
    assert manifest.paths[h2_key].name == "H2-H2_Borysow2001.cia"
    assert manifest.config["cia"]["H2-H2"]["asset"] == h2_key
    assert manifest.config["cia"]["H2-He"]["asset"] == he_key
    assert manifest.config["cia"]["H2-H2"]["metadata"]["temperature_range_K"] == [1000., 7000.]
    assert "H2-H2_Borysow2001_raw" in manifest.sha256
    assert "cia_preparation" in manifest.sha256
