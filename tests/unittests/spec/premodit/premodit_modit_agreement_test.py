""" This test checks the agreement between PreMODIT and MODIT within 1% accuracy.
"""

import pytest
import jax.numpy as jnp
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opacalc import OpaModit
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid

@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_agreement_open_premodit_modit(diffmode):
    """test agreement between PreMODIT and MODIT (zeroscan) for open aliasing"""
    from jax import config

    config.update("jax_enable_x64", True)
    mdb = mock_mdbExomol()
    nus, wav, res = mock_wavenumber_grid()
    Ttest = 1200.0
    P = 1.0
    # PreMODIT LSD
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nus, auto_trange=[1000.0, 1500.0], diffmode=diffmode, alias="open", cutwing=0.5
    )
    xsv = opa.xsvector(Ttest, P)
    opa_modit = OpaModit(mdb, nus, alias="open", cutwing=0.5)
    xsv_modit = opa_modit.xsvector(Ttest, P)
    #dxsv = jnp.abs(xsv_modit / xsv - 1)
    #maxdiff = jnp.max(dxsv)
    #print("maximum differnce = ", maxdiff)

    #assert maxdiff < 0.01
    # maximum differnce =  0.0058, 0.004, 0.008, for diffmode=0,1,2 2/4 2025 @manbou


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_agreement_premodit_modit(diffmode):
    """test agreement between PreMODIT and MODIT (zeroscan) for close aliasing"""
    from jax import config

    config.update("jax_enable_x64", True)
    mdb = mock_mdbExomol()
    nus, wav, res = mock_wavenumber_grid()
    Ttest = 1200.0
    P = 1.0
    # PreMODIT LSD
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nus, auto_trange=[1000.0, 1500.0], diffmode=diffmode
    )
    xsv = opa.xsvector(Ttest, P)
    opa_modit = OpaModit(mdb, nus)
    xsv_modit = opa_modit.xsvector(Ttest, P)
    dxsv = jnp.abs(xsv_modit / xsv - 1)
    maxdiff = jnp.max(dxsv)
    print("maximum differnce = ", maxdiff)

    assert maxdiff < 0.01
    # maximum differnce =  0.0058, 0.004, 0.008, for diffmode=0,1,2 2/4 2025 @manbou


if __name__ == "__main__":
    test_agreement_open_premodit_modit(diffmode=0)
    #test_agreement_open_premodit_modit(diffmode=1)
    #test_agreement_open_premodit_modit(diffmode=2)
    #test_agreement_premodit_modit(diffmode=0)
    #test_agreement_premodit_modit(diffmode=1)
    #test_agreement_premodit_modit(diffmode=2)
