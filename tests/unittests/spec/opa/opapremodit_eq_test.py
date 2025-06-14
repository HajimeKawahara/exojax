from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.opacity import OpaPremodit
from exojax.utils.grids import wavenumber_grid
import copy


def test_opapremodit_eq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True)
    opa = copy.deepcopy(opa_orig)

    assert opa == opa_orig

def test_opapremodit_neq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[300.0,1000.0])
    opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[300.0,1200.0])
    
    assert opa != opa_orig


if __name__ == "__main__":
    test_opapremodit_eq()
    test_opapremodit_neq()
