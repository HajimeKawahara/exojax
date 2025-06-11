""" This test checks the agreement between OpaPremodit and PreMODIT manual execution
"""

import pytest
import numpy as np
import jax.numpy as jnp
from exojax.opacity.opacalc import OpaPremodit
from exojax.opacity.premodit import unbiased_lsd_zeroth
from exojax.opacity.premodit import unbiased_lsd_first
from exojax.opacity.premodit import unbiased_lsd_second
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.opacity.profconv import calc_xsection_from_lsd_scanfft
from exojax.opacity.premodit import unbiased_ngamma_grid
from exojax.database.hitran import normalized_doppler_sigma


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_premodit_opa_and_manual_agreement(diffmode):
    """Test the agreement between the automatic computation of cross section by xsvector 
    and manual computation of cross section by calc_xsection_from_lsd_scanfft. for Premodit
    """
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
    (
        multi_index_uniqgrid,
        elower_grid,
        ngamma_ref_grid,
        n_Texp_grid,
        R,
        pmarray,
    ) = opa.opainfo

    # automatic computing by xsvector
    xsv = opa.xsvector(Ttest, P)
    # tries manual computation of xsvector below
    qt = mdb.qr_interp(Ttest, opa.Tref)
    if diffmode == 0:
        Slsd_premodit = unbiased_lsd_zeroth(
            opa.lbd_coeff[0], Ttest, opa.Tref, nus, elower_grid, qt
        )
    elif diffmode == 1:
        Slsd_premodit = unbiased_lsd_first(
            opa.lbd_coeff, Ttest, opa.Tref, opa.Twt, opa.nu_grid, elower_grid, qt
        )
    elif diffmode == 2:
        Slsd_premodit = unbiased_lsd_second(
            opa.lbd_coeff, Ttest, opa.Tref, opa.Twt, opa.nu_grid, elower_grid, qt
        )
    nsigmaD = normalized_doppler_sigma(Ttest, mdb.molmass, R)
    ngamma_grid = unbiased_ngamma_grid(
        Ttest,
        P,
        ngamma_ref_grid,
        n_Texp_grid,
        multi_index_uniqgrid,
        opa.Tref_broadening,
    )
    log_ngammaL_grid = jnp.log(ngamma_grid)
    xsv_manual = calc_xsection_from_lsd_scanfft(
        Slsd_premodit, R, pmarray, nsigmaD, nus, log_ngammaL_grid
    )

    dxsv = jnp.abs(xsv_manual / xsv - 1)
    assert np.max(dxsv) < 1.0e-11  # (several x e-12, feb 4th 2025)


if __name__ == "__main__":
    test_premodit_opa_and_manual_agreement(diffmode=0)
    test_premodit_opa_and_manual_agreement(diffmode=1)
    test_premodit_opa_and_manual_agreement(diffmode=2)
