import pytest
from exojax.spec.molinfo import mean_molmass
from exojax.utils.isotopes import molmass_hitran

def isotope_molmass(simple_molecule_name):
    mean_molmass, molmass_isotope, abundance_isotope = molmass_hitran()
    
    if simple_molecule_name == 'air' or simple_molecule_name == 'Air':
        return 28.97

def test_isotope_molmass():
    isotope_molmass("CO",1)

def test_molmass():
    assert mean_molmass("CO") == pytest.approx(28.01044518292034)

if __name__ == "__main__":
    test_molmass()
    #test_read_HITRAN_molparam()