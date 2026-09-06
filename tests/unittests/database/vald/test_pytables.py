"""Tests for the dependency-safe PyTables VALD cache backend."""

import gzip
import inspect

import numpy as np
import pandas as pd
import pytest

from exojax.database.core_atom.io import read_ExAll
from exojax.database.vald.api import AdbVald, _load_vald_dataframe


_VALD_FIXTURE = """                                                                    Lande factors       Damping parameters
Elm Ion      WL_vac(A)   log gf* E_low(eV) J lo E_up(eV)  J up  lower   upper    mean    Rad.  Stark  Waals
'Li 1',      4999.0000,  -1.000,  1.0000,  1.0,  3.0000,  2.0,  0.000,  0.000,  0.000,  8.000, -6.000, -7.000,
'Fe 1',      5000.0000,  -1.000,  1.0000,  1.0,  3.0000,  2.0,  0.000,  0.000,  0.000,  8.000, -6.000, -7.000,
'Fe 2',      5001.0000,  -1.000,  1.0000,  1.0,  3.0000,  2.0,  0.000,  0.000,  0.000,  8.000, -6.000, -7.000,
"""


def _write_vald_fixture(path):
    with gzip.open(path, "wt") as stream:
        stream.write(_VALD_FIXTURE)


def test_adbvald_defaults_to_pytables():
    signature = inspect.signature(AdbVald)
    assert signature.parameters["engine"].default == "pytables"


@pytest.mark.parametrize("use_cache", [False, True], ids=["raw", "cache"])
def test_adbvald_preserves_species(tmp_path, use_cache):
    vald_path = tmp_path / "vald_fixture.gz"
    _write_vald_fixture(vald_path)

    if use_cache:
        _load_vald_dataframe(vald_path, engine="pytables")
        vald_path.unlink()

    adb = AdbVald(vald_path, nurange=[19990.0, 20010.0])

    assert adb.engine == "pytables"
    assert vald_path.with_suffix(".h5").is_file()
    np.testing.assert_allclose(
        adb.nu_lines, 1.0e8 / np.array([5001.0, 5000.0, 4999.0])
    )
    np.testing.assert_array_equal(adb.ielem, [26, 26, 3])
    np.testing.assert_array_equal(adb.iion, [2, 1, 1])
    assert adb.pfdat.iloc[np.asarray(adb.QTmask)]["T[K]"].tolist() == [
        "Fe_II",
        "Fe_I",
        "Li_I",
    ]


def test_pandas_alias_creates_and_reuses_pytables_cache(tmp_path):
    vald_path = tmp_path / "vald_fixture.gz"
    _write_vald_fixture(vald_path)

    initial = _load_vald_dataframe(vald_path, engine="pandas")
    cache_path = vald_path.with_suffix(".h5")

    assert cache_path.is_file()
    assert initial["species"].tolist() == [300.0, 2600.0, 2601.0]
    assert initial["wav_lines"].tolist() == [4999.0, 5000.0, 5001.0]

    vald_path.unlink()
    cached = _load_vald_dataframe(vald_path, engine="pytables")
    pd.testing.assert_frame_equal(cached, initial)


def test_read_exall_rejects_unknown_engine(tmp_path):
    with pytest.raises(ValueError, match="Unsupported VALD engine"):
        read_ExAll(tmp_path / "unused.gz", engine="not-an-engine")
