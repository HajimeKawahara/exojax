from exojax.utils.isotopes import get_isotope
from exojax.utils.isotopes import get_stable_isotope
from exojax.utils.isodata import read_mnlist
from exojax.utils.isotopes import molmass_hitran
import numpy as np
import pytest


def test_molarmass_hitran():
    molmass_isotope, abundance_isotope = molmass_hitran()
    assert molmass_isotope["CO"][1] == 27.994915
    assert molmass_isotope["CO"][0] == pytest.approx(28.01044518292034)  # mean
    assert abundance_isotope["CO"][1] == pytest.approx(9.86544e-01)


def test_get_isotope():
    isolist = read_mnlist()
    ref = (
        ["1H", "2H", "3H"],
        [1.007825, 2.014102, 3.016049],
        [99.9885, 0.0115, np.nan],
    )
    assert np.all(get_isotope("H", isolist)[0:2] == ref[0:2])
    assert get_isotope("H", isolist)[2][2] != get_isotope("H", isolist)[2][2]
    assert np.all(np.array(get_isotope("H", isolist)[2][0:2]) == ref[2][0:2])


def test_get_stable_isotope():
    isolist = read_mnlist()
    ref = ("1H", 1.007825, 99.9885)
    assert np.all(get_stable_isotope("H", isolist) == ref)


if __name__ == "__main__":
    test_get_isotope()
    test_get_stable_isotope()
    test_molarmass_hitran()
