from exojax.spec.opacalc import OpaPremodit
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
import numpy as np
import jax.numpy as jnp
import pytest


def test_OpaPremodit_Stitch_initialization():
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    opas = OpaPremodit(mdb, nus, nstitch=4, auto_trange=[500,1300], allow_32bit=True)
    
    
def test_OpaPremodit_Stitch_check_nu_grid_reducible_raise_error():
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    with pytest.raises(ValueError):
        opas = OpaPremodit(mdb, nus, nstitch=3, allow_32bit=True)


def test_OpaPremoditStitch_xsv_agreement_Premodit(fig=False):
    from jax import config 
    config.update("jax_enable_x64", True)
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    ndiv = 4    
    opas = OpaPremodit(mdb, nus, nstitch=ndiv, auto_trange=[500,1300], cutwing = 0.125)
    opa = OpaPremodit(mdb, nus, auto_trange=[500,1300], alias="open", cutwing = 0.125)
    xsv_s = opas.xsvector(1000.0, 1.0)
    xsv = opa.xsvector(1000.0, 1.0)
    xsv_trim = xsv[opa.filter_length_oneside:-opa.filter_length_oneside]
    diff = xsv_s/xsv_trim-1.0
    print(np.max(np.abs(diff)))
    assert np.max(np.abs(diff)) < 2.e-11 #1.02e-11 3/1 2025 opapremodit

    if fig:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(211)
        plt.plot(nus, xsv_s,label="stitch", ls="dashed")
        plt.plot(nus, xsv_trim,label="premodit",alpha=0.3)
        plt.yscale("log")
        ax = fig.add_subplot(212)
        plt.plot(nus, diff)
        plt.show()
        
def test_OpaPremoditStitch_xsm_agreement_Premodit():
    from jax import config 
    config.update("jax_enable_x64", True)
    nus, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    
    ndiv = 4    
    opas = OpaPremodit(mdb, nus, nstitch=ndiv, auto_trange=[500,1300], cutwing = 0.125)
    opa = OpaPremodit(mdb, nus, auto_trange=[500,1300], alias="open", cutwing = 0.125)
    Tarr = jnp.array([1000.0, 1100.0])
    Parr = jnp.array([1.0, 1.5])
    xsm_s = opas.xsmatrix(Tarr, Parr)
    xsm = opa.xsmatrix(Tarr, Parr)
    xsm_trim = xsm[:, opa.filter_length_oneside:-opa.filter_length_oneside]
    diff = xsm_s/xsm_trim-1.0
    print(np.max(np.abs(diff)))
    assert np.max(np.abs(diff)) < 2.e-11 # 1.025e-11 3/1 2025 opapremodit


if __name__ == "__main__":
    #test_OpaPremodit_Stitch_initialization()
    #test_OpaPremodit_Stitch_check_nu_grid_reducible_raise_error()
    test_OpaPremoditStitch_xsv_agreement_Premodit(fig=False)
    test_OpaPremoditStitch_xsm_agreement_Premodit()
    # 

