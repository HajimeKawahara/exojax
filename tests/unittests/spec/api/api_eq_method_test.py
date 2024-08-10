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

def test_neq_Hitemp():
    mdb_orig = mock_mdbHitemp(multi_isotope=True)
    mdb  = copy.deepcopy(mdb_orig)
    mdb.change_reference_temperature(1320.0)
    assert mdb_orig != mdb

def test_neq_Exomol():
    mdb_orig = mock_mdbExomol()
    mdb  = copy.deepcopy(mdb_orig)
    mdb.change_reference_temperature(1320.0)
    assert mdb_orig != mdb


if __name__ == "__main__":
    #test_eq_Hitemp()
    #test_eq_Exomol()
    #test_neq_Hitemp()
    test_neq_Exomol()