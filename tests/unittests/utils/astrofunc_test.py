from exojax.utils.astrofunc import getjov_gravity
from exojax.utils.astrofunc import getjov_logg
import pytest

def test_getjov_logg():
    logg = getjov_logg(1.0,1.0)
    assert logg == pytest.approx(3.3942025)
    
def test_getjov_gravity():
    g = getjov_gravity(1.0,1.0)
    assert g == pytest.approx(2478.57730044555)
    
if __name__ == "__main__":
    test_getjov_logg()