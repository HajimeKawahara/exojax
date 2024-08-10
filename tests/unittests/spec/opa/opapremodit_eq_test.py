from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.spec.opacalc import OpaPremodit
import copy

def test_eq_Hitemp():
    mdb_orig  = copy.deepcopy(mock_mdbHitemp(multi_isotope=True))
    mdb = mock_mdbHitemp(multi_isotope=True)
    assert mdb_orig == mdb
