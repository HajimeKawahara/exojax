from exojax.atm.psat import psat_water_Magnus

def test_psat_water():
    psat = psat_water_Magnus(100.0)
    assert psat == 1.040767


if __name__ == "__main__":
    test_psat_water()
