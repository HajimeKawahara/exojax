"""This test checks the agreement between MODIT open and close alising for zeroscan calculation"""

import jax.numpy as jnp
from exojax.database.core.line_strength import line_strength
from exojax.opacity._common.set_ditgrid import ditgrid_log_interval
from exojax.database.core.broadening import gamma_exomol
from exojax.database.core.broadening import normalized_doppler_sigma
from exojax.opacity.initspec import init_modit
from exojax.opacity.modit.modit import xsvector_open_zeroscan
from exojax.opacity.modit.modit import xsvector_zeroscan
from exojax.opacity import OpaModit
from exojax.rt import ArtEmisPure
from exojax.utils.grids import extended_wavenumber_grid
from exojax.utils.constants import Tref_original
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid

from jax import config

config.update("jax_enable_x64", True)


def test_open_close_xsmatrix_modit_agreement(db="exomol"):
    nu_grid, _, _ = mock_wavenumber_grid()
    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=2, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mdb = mock_mdb(db)
    opa_close = OpaModit(
        mdb=mdb,
        nu_grid=nu_grid,
        Tarr_list=Tarr,
        Parr=art.pressure,
        dit_grid_resolution=0.2,
        alias="close",
    )
    xsmatrix_close = opa_close.xsmatrix(Tarr, art.pressure)
    opa_open = OpaModit(
        mdb=mdb,
        nu_grid=nu_grid,
        Tarr_list=Tarr,
        Parr=art.pressure,
        dit_grid_resolution=0.2,
        alias="open",
        cutwing=1.0,
    )
    xsmatrix_open = opa_open.xsmatrix(Tarr, art.pressure)

    diff = (
        xsmatrix_close
        / xsmatrix_open[
            :, opa_open.filter_length_oneside : -opa_open.filter_length_oneside
        ]
        - 1.0
    )
    maxdiff = jnp.max(jnp.abs(diff))
    assert maxdiff < 0.006


def test_agreement_open_and_close_zeroscan_modit():
    """test agreement between scanfft and zeroscan calculation for MODIT"""
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

    xsv_zeroscan_close = xsvector_zeroscan(
        cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid
    )

    nextend = len(nus)
    nu_grid_extended = extended_wavenumber_grid(nus, nextend, nextend)
    xsv_zeroscan_open = xsvector_open_zeroscan(
        cont_nu,
        index_nu,
        R,
        nsigmaD,
        ngammaL,
        Sij,
        nus,
        ngammaL_grid,
        nu_grid_extended,
        nextend,
    )

    diff = xsv_zeroscan_close / xsv_zeroscan_open[nextend:-nextend] - 1.0
    assert jnp.max(jnp.abs(diff)) < 1.0e-4  # 2.0011695423982623e-05 Feb. 17th 2025
