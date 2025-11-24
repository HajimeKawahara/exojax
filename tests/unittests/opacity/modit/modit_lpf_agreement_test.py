""" This test checks the agreement between MODIT and LPF 
"""

import jax.numpy as jnp
from exojax.opacity import OpaDirect
from exojax.opacity import OpaModit
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid


def test_agreement_modit_lpf():
    """test agreement between MODIT and LPF DIRECT"""
    from jax import config

    config.update("jax_enable_x64", True)
    mdb = mock_mdbExomol()
    nus, wav, res = mock_wavenumber_grid()
    Ttest = 1200.0
    P = 1.0

    opa = OpaDirect(mdb, nus)
    xsv_lpf = opa.xsvector(Ttest, P)

    opamodit = OpaModit(mdb, nus)
    xsv_modit = opamodit.xsvector(Ttest, P)

    dxsv = jnp.abs(xsv_lpf / xsv_modit - 1)
    maxdiff = jnp.max(dxsv)
    print("maximum differnce = ", maxdiff)
    assert maxdiff < 0.0008  # 0.0007584222072695157


if __name__ == "__main__":
    test_agreement_modit_lpf()
