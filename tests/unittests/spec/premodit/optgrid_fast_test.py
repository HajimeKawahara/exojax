""" unittest for exojax.spec.premodit.optgrid

Note: 
    for the complete test, use integration/unittests_long/premodit/optgrid_test.py

"""


from exojax.opacity.premodit.optgrid import optelower
import pytest
from jax import config
config.update("jax_enable_x64", False)


def test_optelower_exomol_fast():
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    
    nu_grid, wav, reso = mock_wavenumber_grid()
    Tmax = 1020.0  #K
    Pmin = 0.1  #bar
    mdb = mock_mdbExomol(crit=1.e-37)
    Eopt = optelower(mdb, nu_grid, Tmax, Pmin, accuracy=0.0)
    print("optimal elower_max=",Eopt,"cm-1")
    assert Eopt == pytest.approx(11615.5075)

if __name__ == "__main__":
    test_optelower_exomol_fast()