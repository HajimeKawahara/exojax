import numpy as np

from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaDirect

def test_opadirect_hitemp_call():
    nus, wav, res = wavenumber_grid(22920.0, 23100.0, 64, unit="AA", xsmode="premodit")
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa = OpaDirect(mdb, nu_grid=nus)
    xsv = opa.xsvector(1000.0, 1.0)
    xsm = opa.xsmatrix(np.array([1000.0, 1200.0]), np.array([1.0, 0.1]))

    assert isinstance(opa, OpaDirect)
    assert xsv.shape == nus.shape
    assert xsm.shape == (2, len(nus))
    assert np.all(np.isfinite(xsv))
    assert np.all(np.isfinite(xsm))
