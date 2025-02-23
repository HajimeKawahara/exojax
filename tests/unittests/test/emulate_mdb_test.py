from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
import numpy as np
import pytest

@pytest.mark.parametrize("molecule", ["CO", "H2O"])
def test_mock_mdbExoMol(molecule):
    mdb = mock_mdbExomol(molecule)
    print(np.sum(mdb.logsij0))
    print(len(mdb.logsij0))
    ref = {"CO": -69819.11, "H2O": -12637.281}
    lenval = {"CO": 259, "H2O": 197}
    assert np.sum(mdb.logsij0) == pytest.approx(ref[molecule])
    assert len(mdb.logsij0) == lenval[molecule]


def test_mock_mdbHitemp():
    mdb = mock_mdbHitemp()
    print(np.sum(mdb.logsij0))
    print(len(mdb.logsij0))
    assert np.sum(mdb.logsij0) == pytest.approx(-70108.27)
    assert len(mdb.logsij0) == 260

    mdb = mock_mdbHitemp(multi_isotope=True)
    print(np.sum(mdb.logsij0))
    print(len(mdb.logsij0))
    assert np.sum(mdb.logsij0) == pytest.approx(-421638.25)
    assert len(mdb.logsij0) == 1368

if __name__ == "__main__":
    test_mock_mdbExoMol("CO")
    test_mock_mdbExoMol("H2O")
    test_mock_mdbHitemp()