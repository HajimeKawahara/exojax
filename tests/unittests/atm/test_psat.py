from exojax.atm.psat import psat_water_Magnus
from exojax.atm.psat import psat_water_AM01
from exojax.utils.constants import Tc_water
import pytest

def test_psat_water():
    psat = psat_water_Magnus(100.0+Tc_water)
    assert psat == pytest.approx(1.040767)
    psat = psat_water_AM01(100.0+Tc_water)
    assert psat == pytest.approx(1.0130779)
