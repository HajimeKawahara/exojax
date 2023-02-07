from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
import numpy as np
import pytest

def test_mock_mdbExoMol():
    mdb = mock_mdbExomol()
    print(np.sum(mdb.logsij0))
    print(len(mdb.logsij0))
    assert np.sum(mdb.logsij0) == pytest.approx(-69819.11)
    assert len(mdb.logsij0) == 259


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
    test_mock_mdbExoMol()
    test_mock_mdbHitemp()