import numpy as np

from exojax.utils.grids import wavenumber_grid
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit

def test_stitching_accepts_mdb_lines_outside_grid():
    """Preserve the grid and stitching parameters that reproduced issue #586."""
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
    xsv_stitch_all = np.asarray(opa.xsvector(T, P))
    assert xsv_stitch_all.shape == nu_grid.shape
    assert np.all(np.isfinite(xsv_stitch_all))
    assert np.all(xsv_stitch_all >= 0.0)
