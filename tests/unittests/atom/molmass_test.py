"""test for molmass calculation"""

import pytest
from exojax.database.molinfo  import molmass_isotope


def test_molmass():
    assert molmass_isotope('air')==28.97
    assert molmass_isotope('CO2')==pytest.approx(43.98983)
    assert molmass_isotope('He')==4.002602
    assert molmass_isotope('CO2',db_HIT=True)==43.98983
    assert molmass_isotope('He',db_HIT=True)==4.002602
    

if __name__ == '__main__':
    test_molmass()
