import numpy as np

from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaModit

def test_opamodit_hitemp_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa = OpaModit(mdb, nu_grid=nus, allow_32bit=True)
    xsv = opa.xsvector(1000.0, 1.0)

    assert isinstance(opa, OpaModit)
    assert xsv.shape == nus.shape
    assert np.all(np.isfinite(xsv))

def test_opamodit_exomol_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbExomol()
    opa = OpaModit(mdb, nu_grid=nus, allow_32bit=True)
    
    assert isinstance(opa, OpaModit)

if __name__ == "__main__":
    test_opamodit_hitemp_call()
    test_opamodit_exomol_call()
