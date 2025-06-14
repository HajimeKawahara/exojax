from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.opacity import OpaModit
from exojax.utils.grids import wavenumber_grid
import copy


def test_opamodit_eq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaModit(mdb, nu_grid=nus, allow_32bit=True)
    opa = copy.deepcopy(opa_orig)

    assert opa == opa_orig

def test_opamodit_neq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaModit(mdb, nu_grid=nus, allow_32bit=True)
    
    nus, wav, res = wavenumber_grid(12920.0, 13100.0, 20000, unit="AA", xsmode="premodit")
    opa = OpaModit(mdb, nu_grid=nus, allow_32bit=True)
    
    assert opa != opa_orig


if __name__ == "__main__":
    test_opamodit_eq()
    test_opamodit_neq()
