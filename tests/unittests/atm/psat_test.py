from exojax.atm.psat import psat_water_Magnus
from exojax.atm.psat import psat_water_AM01
from exojax.utils.constants import Tc_water
import pytest

def test_psat_water():
    psat = psat_water_Magnus(100.0+Tc_water)
    assert psat == pytest.approx(1.040767)
    psat = psat_water_AM01(100.0+Tc_water)
    assert psat == pytest.approx(1.0130779)
    
import matplotlib.pyplot as plt
import numpy as np
def comparison_water():
    t = np.logspace(2,3,300)
    plt.plot(t,psat_water_Magnus(t),label="Magnus")
    plt.plot(t,psat_water_AM01(t),label="AM01 (Buck 81,96)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("vapor pressure (bar)")
    plt.xlabel("temperature (K)")
    plt.show()


if __name__ == "__main__":
    test_psat_water()
    comparison_water()


