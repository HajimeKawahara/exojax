from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp


def test_mock_mdbExoMol():
    mdb = mock_mdbExomol()
    #assert np.sum(mdb.logsij0) == pytest.approx(-11840.455)
    #assert len(mdb.logsij0) == 192


def test_mock_mdbHitemp():
    mdb = mock_mdbHitemp()
    #assert np.sum(mdb.logsij0) == pytest.approx(-11762.4795)
    #assert len(mdb.logsij0) == 191
    mdb = mock_mdbHitemp(multi_isotope=True)
    #assert np.sum(mdb.logsij0) == pytest.approx(-33647.324)
    #assert len(mdb.logsij0) == 525


if __name__ == "__main__":
    test_mock_mdbExoMol()
    test_mock_mdbHitemp()