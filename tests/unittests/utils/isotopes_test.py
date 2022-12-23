from exojax.utils.isotopes import get_isotope
from exojax.utils.isotopes import get_stable_isotope
from exojax.utils.isotopes import isodata
from exojax.utils.isotopes import exact_hitran_isotope_name_from_isotope
from exojax.utils.isotopes import molmass_hitran
import numpy as np
import pytest

def exact_molecule_name_to_isotope_number(exact_molecule_name):
    from radis.db.molparam import isotope_name_dict

    #check exomol exact name
    keys = [k for k, v in isotope_name_dict.items() if v == exact_molecule_name]
    if len(keys) == 0:
        #check hitran exact name
        exact_hitran_molecule_name = exact_molname_exomol_to_hitran(exact_molecule_name)
        keys = [k for k, v in isotope_name_dict.items() if v == exact_hitran_molecule_name]
    if len(keys) == 1:
        isotope_number = keys[0][1]
    else:
        raise ValueError("No isotope number identified.")

    return isotope_number

def exact_molname_exomol_to_hitran(exact_exomol_molecule_name):
    return "("+exact_exomol_molecule_name.replace("-",")(")+")"

def test_exact_exomol_molecule_name_to_isotope_number():
    eemn="12C-16O"
    isonum=exact_molecule_name_to_isotope_number(eemn)
    assert isonum==0
    eemn="16O-13C-17O"
    isonum=exact_molecule_name_to_isotope_number(eemn)
    assert isonum==6
    
def test_exact_molname_exomol_to_hitran():
    eemn="16O-13C-17O"
    ehmn=exact_molname_exomol_to_hitran(eemn)
    assert ehmn == "(16O)(13C)(17O)"

def test_molarmass_hitran():
    molmass_isotope, abundance_isotope = molmass_hitran()
    assert molmass_isotope["CO"][1] == 27.994915
    assert molmass_isotope["CO"][0] == pytest.approx(28.01044518292034) #mean
    assert abundance_isotope["CO"][1] == pytest.approx(9.86544E-01)


def test_exact_isotope_name_from_isotope():
    simple_molecule_name = "CO"
    isotope = 1
    assert exact_hitran_isotope_name_from_isotope(simple_molecule_name,
                                           isotope) == "(12C)(16O)"

    simple_molecule_name = "H2O"
    isotope = 5
    assert exact_hitran_isotope_name_from_isotope(simple_molecule_name, isotope) == "HD(18O)"


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
    test_exact_molname_exomol_to_hitran()
    test_exact_exomol_molecule_name_to_isotope_number()