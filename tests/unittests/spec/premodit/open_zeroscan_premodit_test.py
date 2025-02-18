""" This test checks the agreement between PreMODIT open and close aliasing
"""

import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.test.emulate_mdb import mock_mdb
from exojax.spec.opacalc import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.atmrt import ArtEmisPure

from jax import config

config.update("jax_enable_x64", True)


def test_open_close_xsmatrix_premodit_agreement(db="exomol"):
    nu_grid, _, _ = mock_wavenumber_grid()
    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mdb = mock_mdb(db)
    opa_close = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        dit_grid_resolution=0.2,
        auto_trange=[400.0, 1500.0],
        alias="close",
    )
    xsmatrix_close = opa_close.xsmatrix(Tarr, art.pressure)
    opa_open = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        dit_grid_resolution=0.2,
        auto_trange=[400.0, 1500.0],
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
    print(maxdiff)  # 0.003954996543941491 Feb. 18th 2025
    assert maxdiff < 0.004


if __name__ == "__main__":
    test_open_close_xsmatrix_premodit_agreement(db="exomol")
