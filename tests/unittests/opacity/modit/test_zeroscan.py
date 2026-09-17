"""This test checks the agreement between MODIT scanfft and zeroscan calculation"""

import jax.numpy as jnp
from exojax.database.core.line_strength import line_strength
from exojax.opacity._common.set_ditgrid import ditgrid_log_interval
from exojax.database.core.broadening import gamma_exomol
from exojax.utils.constants import Tref_original
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.database.core.broadening import normalized_doppler_sigma
from exojax.opacity.initspec import init_modit
from exojax.opacity.modit.modit import xsvector_scanfft
from exojax.opacity.modit.modit import xsvector_zeroscan


def test_agreement_scanfft_zeroscan_modit():
    """test agreement between scanfft and zeroscan calculation"""
    from jax import config

    config.update("jax_enable_x64", True)
    mdb = mock_mdbExomol()
    nus, wav, res = mock_wavenumber_grid()
    Ttest = 1200.0
    P = 1.0

    # MODIT manual
    # We need to revert the reference temperature to 296K to reuse mdb for MODIT

    # mdb.change_reference_temperature(Tref_original)
    qt = mdb.qr_interp(Ttest, Tref_original)
    cont, index, R, pmarray = init_modit(mdb.nu_lines, nus)
    nsigmaD = normalized_doppler_sigma(Ttest, mdb.molmass, R)
    Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
    gammaL = gamma_exomol(P, Ttest, mdb.n_Texp, mdb.alpha_ref)

    dv_lines = mdb.nu_lines / R
    ngammaL = gammaL / dv_lines
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.2)
    ## also, xs
    Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
    cont_nu, index_nu, R, pmarray = init_modit(mdb.nu_lines, nus)
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.2)

    xsv_scanfft = xsvector_scanfft(
        cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid
    )
    xsv_zeroscan = xsvector_zeroscan(
        cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid
    )

    dxsv = jnp.abs(xsv_scanfft / xsv_zeroscan - 1)
    maxdiff = jnp.max(dxsv)
    assert maxdiff < 2.0e-12  # 1.5e-12
