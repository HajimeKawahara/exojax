"""Tests for the dependency-safe PyTables VALD cache backend."""

import gzip
import inspect

import pandas as pd
import pytest

from exojax.database.core_atom.io import read_ExAll
from exojax.database.vald.api import AdbVald, _load_vald_dataframe


_VALD_FIXTURE = """                                                                    Lande factors       Damping parameters
Elm Ion      WL_vac(A)   log gf* E_low(eV) J lo E_up(eV)  J up  lower   upper    mean    Rad.  Stark  Waals
'Fe 1',      5000.0000,  -1.000,  1.0000,  1.0,  3.0000,  2.0,  0.000,  0.000,  0.000,  8.000, -6.000, -7.000,
"""


def _write_vald_fixture(path):
    with gzip.open(path, "wt") as stream:
        stream.write(_VALD_FIXTURE)


def test_adbvald_defaults_to_pytables():
    signature = inspect.signature(AdbVald)
    assert signature.parameters["engine"].default == "pytables"


def test_adbvald_loads_default_pytables_cache(tmp_path):
    vald_path = tmp_path / "vald_fixture.gz"
    _write_vald_fixture(vald_path)

    adb = AdbVald(vald_path, nurange=[19999.0, 20001.0])

    assert adb.engine == "pytables"
    assert len(adb.nu_lines) == 1
    assert vald_path.with_suffix(".h5").is_file()


def test_pandas_alias_creates_and_reuses_pytables_cache(tmp_path):
    vald_path = tmp_path / "vald_fixture.gz"
    _write_vald_fixture(vald_path)

    initial = _load_vald_dataframe(vald_path, engine="pandas")
    cache_path = vald_path.with_suffix(".h5")

    assert cache_path.is_file()
    assert initial["species"].tolist() == [2600.0]
    assert initial["wav_lines"].tolist() == [5000.0]

    vald_path.unlink()
    cached = _load_vald_dataframe(vald_path, engine="pytables")
    pd.testing.assert_frame_equal(cached, initial)


def test_read_exall_rejects_unknown_engine(tmp_path):
    with pytest.raises(ValueError, match="Unsupported VALD engine"):
        read_ExAll(tmp_path / "unused.gz", engine="not-an-engine")
