from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaPremodit

def test_opapremodit_hitemp_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True)

    assert isinstance(opa, OpaPremodit)

def test_opapremodit_exomol_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbExomol()
    opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True)
    
    assert isinstance(opa, OpaPremodit)

if __name__ == "__main__":
    test_opapremodit_hitemp_call()
    test_opapremodit_exomol_call()
