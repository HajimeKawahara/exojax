import numpy as np

from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaDirect

def test_opadirect_hitemp_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa = OpaDirect(mdb, nu_grid=nus)
    xsv = opa.xsvector(1000.0, 1.0)
    xsm = opa.xsmatrix(np.array([1000.0, 1200.0]), np.array([1.0, 0.1]))

    assert isinstance(opa, OpaDirect)
    assert xsv.shape == nus.shape
    assert xsm.shape == (2, len(nus))
    assert np.all(np.isfinite(xsv))
    assert np.all(np.isfinite(xsm))

def test_opadirect_exomol_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 20000, unit="AA", xsmode="premodit")
    mdb = mock_mdbExomol()
    opa = OpaDirect(mdb, nu_grid=nus)
    
    assert isinstance(opa, OpaDirect)

if __name__ == "__main__":
    test_opadirect_hitemp_call()
    test_opadirect_exomol_call()
