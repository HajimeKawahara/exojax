from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
import copy

def test_eq_Hitemp():
    mdb_orig  = copy.deepcopy(mock_mdbHitemp(multi_isotope=True))
    mdb = mock_mdbHitemp(multi_isotope=True)
    assert mdb_orig == mdb

def test_eq_Exomol():
    mdb_orig  = copy.deepcopy(mock_mdbExomol())
    mdb = mock_mdbExomol()
    assert mdb_orig == mdb



if __name__ == "__main__":
    test_eq_Hitemp()
    test_eq_Exomol()
    