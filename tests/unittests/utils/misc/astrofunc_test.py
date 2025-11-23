from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.astrofunc import logg_jupiter
from exojax.utils.astrofunc import square_radius_from_mass_logg
import pytest

def test_square_radius_from_mass_logg():

    Mp = 2.0
    logg = 4.0
    Rp2 = square_radius_from_mass_logg(Mp, logg)
    assert Rp2 == pytest.approx(0.4957154)
    

def test_getjov_logg():
    logg = logg_jupiter(1.0,1.0)
    assert logg == pytest.approx(3.3942025)
    
def test_getjov_gravity():
    g = gravity_jupiter(1.0,1.0)
    assert g == pytest.approx(2478.57730044555)
    
if __name__ == "__main__":
    test_getjov_logg()