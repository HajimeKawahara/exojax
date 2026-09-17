from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.test.data import get_testdata_filename, TESTDATA_CO_HITEMP_PARFILE
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("molecule", ["CO", "H2O"])
def test_mock_mdb_exomol_preserves_existing_directory(molecule, tmp_path):
    existing = tmp_path / molecule
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("existing database")
    mdb = mock_mdbExomol(molecule)
    ref = {"CO": -69819.11, "H2O": -12637.281}
    lenval = {"CO": 259, "H2O": 197}
    assert np.sum(mdb.logsij0) == pytest.approx(ref[molecule])
    assert len(mdb.logsij0) == lenval[molecule]
    assert sentinel.read_text() == "existing database"


def test_mock_mdb_hitemp_copies_source_parfile(tmp_path):
    mdb = mock_mdbHitemp()
    assert np.sum(mdb.logsij0) == pytest.approx(-70108.27)
    assert len(mdb.logsij0) == 260

    mdb = mock_mdbHitemp(multi_isotope=True)
    assert np.sum(mdb.logsij0) == pytest.approx(-421638.25)
    assert len(mdb.logsij0) == 1368

    source = Path(get_testdata_filename(TESTDATA_CO_HITEMP_PARFILE))
    copies = list(tmp_path.glob(f"exojax-hitemp-*/{source.name}"))
    assert len(copies) == 2
    assert all(copy.read_bytes() == source.read_bytes() for copy in copies)
