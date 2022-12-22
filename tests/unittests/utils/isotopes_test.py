from exojax.utils.isotopes import get_isotope
from exojax.utils.isotopes import get_stable_isotope
from exojax.utils.isotopes import isodata
from exojax.utils.isotopes import exact_isotope_name_from_isotope
from exojax.utils.isotopes import molarmass_hitran
import numpy as np
import pytest


def test_molarmass_hitran():
    mean_molmass, molmass_isotope, abundance_isotope = molarmass_hitran()
    assert molmass_isotope["CO"][0] == 27.994915
    assert mean_molmass["CO"] == pytest.approx(28.01044518292034)
    assert abundance_isotope["CO"][0] == pytest.approx(9.86544E-01)


def test_exact_isotope_name_from_isotope():
    simple_molecule_name = "CO"
    isotope = 1
    assert exact_isotope_name_from_isotope(simple_molecule_name,
                                           isotope) == "(12C)(16O)"

    simple_molecule_name = "H2O"
    isotope = 5
    assert exact_isotope_name_from_isotope(simple_molecule_name, isotope) == "HD(18O)"


def test_get_isotope():
    isolist = isodata.read_mnlist()
    ref = (['1H', '2H', '3H'], [1.007825, 2.014102,
                                3.016049], [99.9885, 0.0115, np.nan])
    assert np.all(get_isotope('H', isolist)[0:2] == ref[0:2])
    assert (get_isotope('H', isolist)[2][2] != get_isotope('H', isolist)[2][2])
    assert np.all(np.array(get_isotope('H', isolist)[2][0:2]) == ref[2][0:2])


def test_get_stable_isotope():
    isolist = isodata.read_mnlist()
    ref = ('1H', 1.007825, 99.9885)
    assert np.all(get_stable_isotope('H', isolist) == ref)


if __name__ == "__main__":
    test_get_isotope()
    test_get_stable_isotope()
    test_exact_isotope_name_from_isotope()
    test_molarmass_hitran()