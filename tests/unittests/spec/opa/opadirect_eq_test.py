from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.opacity import OpaDirect
from exojax.utils.grids import wavenumber_grid
import copy


def test_opadirect_eq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaDirect(mdb, nu_grid=nus)
    opa = copy.deepcopy(opa_orig)

    assert opa == opa_orig

def test_opadirect_neq():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa_orig = OpaDirect(mdb, nu_grid=nus)
    
    nus, wav, res = wavenumber_grid(12920.0, 13100.0, 20000, unit="AA", xsmode="premodit")
    opa = OpaDirect(mdb, nu_grid=nus)
    
    assert opa != opa_orig


if __name__ == "__main__":
    test_opadirect_eq()
    test_opadirect_neq()
