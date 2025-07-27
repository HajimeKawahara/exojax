from jax import config

config.update("jax_enable_x64", True)
from exojax.utils.grids import wavenumber_grid
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit

def test_opapremodit_exomol_call():
    """this test was used to identify #586. Leave here, just in case"""
    mdb = mock_mdb("exomol")
    filter_length_oneside = 10000
    nu_grid, _, _ = wavenumber_grid(4325.0, 4365.0, 2*filter_length_oneside, xsmode="premodit")

    T=1000.0 #K
    P=1.0 #bar

    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        dit_grid_resolution=0.2,
        auto_trange=[400.0, 1500.0],
        nstitch=2,
        cutwing=0.5,
    )
    xsv_stitch_all= opa.xsvector(T,P)
    assert True
