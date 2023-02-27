from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.astrofunc import logg_jupiter
import pytest

def test_getjov_logg():
    logg = logg_jupiter(1.0,1.0)
    assert logg == pytest.approx(3.3942025)
    
def test_getjov_gravity():
    g = gravity_jupiter(1.0,1.0)
    assert g == pytest.approx(2478.57730044555)
    
if __name__ == "__main__":
    test_getjov_logg()