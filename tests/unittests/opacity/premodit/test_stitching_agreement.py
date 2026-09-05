from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
import numpy as np
import jax.numpy as jnp
import pytest


def test_stitching_rejects_indivisible_grid():
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    with pytest.raises(ValueError):
        opas = OpaPremodit(mdb, nus, nstitch=3, allow_32bit=True)


def test_stitched_vector_matches_unstitched_opacity():
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    ndiv = 4    
    opas = OpaPremodit(mdb, nus, nstitch=ndiv, auto_trange=[500,1300], cutwing = 1.0)
    opa = OpaPremodit(mdb, nus, auto_trange=[500,1300])
    xsv_s = opas.xsvector(1000.0, 1.0)
    xsv = opa.xsvector(1000.0, 1.0)
    diff = xsv_s/xsv-1.0
    print(np.max(np.abs(diff)))
    assert np.max(np.abs(diff)) < 3.e-5 
    #diff is mainly caused from the diff between lpffilter and analytic expression of Voigt

def test_stitched_matrix_matches_unstitched_opacity():
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    
    ndiv = 4    
    opas = OpaPremodit(mdb, nus, nstitch=ndiv, auto_trange=[500,1300], cutwing = 1.0)
    opa = OpaPremodit(mdb, nus, auto_trange=[500,1300])
    Tarr = jnp.array([1000.0, 1100.0])
    Parr = jnp.array([1.0, 1.5])
    xsm_s = opas.xsmatrix(Tarr, Parr)
    xsm = opa.xsmatrix(Tarr, Parr)
    diff = xsm_s/xsm-1.0
    print(np.max(np.abs(diff)))
    assert np.max(np.abs(diff)) < 3.e-5 


    #
