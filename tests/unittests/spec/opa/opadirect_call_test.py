from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaDirect

def test_opadirect_hitemp_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa = OpaDirect(mdb, nu_grid=nus)

    assert isinstance(opa, OpaDirect)

def test_opadirect_exomol_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbExomol()
    opa = OpaDirect(mdb, nu_grid=nus)
    
    assert isinstance(opa, OpaDirect)

if __name__ == "__main__":
    test_opadirect_hitemp_call()
    test_opadirect_exomol_call()
