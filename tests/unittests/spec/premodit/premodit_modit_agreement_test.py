""" This test checks the agreement between PreMODIT and MODIT within 1% accuracy.
"""

import pytest
import jax.numpy as jnp
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.hitran import line_strength
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.spec.exomol import gamma_exomol
from exojax.utils.constants import Tref_original
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec import normalized_doppler_sigma
from exojax.spec.modit import xsvector_scanfft
from exojax.spec.initspec import init_modit


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_agreement_premodit_modit(diffmode):
    """test agreement between PreMODIT and MODIT"""
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
        lbd_coeff,
        multi_index_uniqgrid,
        elower_grid,
        ngamma_ref_grid,
        n_Texp_grid,
        R,
        pmarray,
    ) = opa.opainfo
    xsv = opa.xsvector(Ttest, P)
    nsigmaD = normalized_doppler_sigma(Ttest, mdb.molmass, R)

    # MODIT LSD
    # We need to revert the reference temperature to 296K to reuse mdb for MODIT

    # mdb.change_reference_temperature(Tref_original)
    qt = mdb.qr_interp(Ttest, Tref_original)
    cont, index, R, pmarray = init_modit(mdb.nu_lines, nus)
    Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
    gammaL = gamma_exomol(P, Ttest, mdb.n_Texp, mdb.alpha_ref)

    dv_lines = mdb.nu_lines / R
    ngammaL = gammaL / dv_lines
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    ## also, xs
    Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
    cont_nu, index_nu, R, pmarray = init_modit(mdb.nu_lines, nus)
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    xsv_modit = xsvector_scanfft(
        cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid
    )

    dxsv = jnp.abs(xsv_modit / xsv - 1)
    maxdiff = jnp.max(dxsv)
    print("maximum differnce = ", maxdiff)
    assert (
        maxdiff < 0.01
    )  # maximum differnce =  0.0058, 0.004, 0.008, for diffmode=0,1,2 2/4 2025 @manbou


if __name__ == "__main__":
    test_agreement_premodit_modit(diffmode=0)
    test_agreement_premodit_modit(diffmode=1)
    test_agreement_premodit_modit(diffmode=2)
