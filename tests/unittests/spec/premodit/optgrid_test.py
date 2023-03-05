from exojax.spec.optgrid import optelower
import pytest
from jax.config import config
config.update("jax_enable_x64", True)


def test_optelower():
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    
    nu_grid, wav, reso = mock_wavenumber_grid()
    Tmax = 1020.0  #K
    Pmin = 0.1  #bar
    mdb = mock_mdbExomol(crit=1.e-37)
    Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
    print("optimal elower_max=",Eopt)
    assert Eopt == pytest.approx(11559.3717)


if __name__ == "__main__":
    test_optelower()