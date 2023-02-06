from exojax.spec.optgrid import optelower
import pytest
from jax.config import config
config.update("jax_enable_x64", True)


def test_optelower():
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.spec import api

    Nx = 20000
    nu_grid, wav, reso = wavenumber_grid(22900.0,
                                         23100.0,
                                         Nx,
                                         unit='AA',
                                         xsmode="modit")
    Tmax = 1020.0  #K
    Pmin = 0.1  #bar
    #mdb = mock_mdbExomol()
    mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',
                          nu_grid,
                          crit=1.e-37,
                          inherit_dataframe=False,
                          gpu_transfer=True)

    Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
    print(Eopt)
    assert Eopt == pytest.approx(11559.3717)


if __name__ == "__main__":
    test_optelower()