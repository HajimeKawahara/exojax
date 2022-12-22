import pytest
from exojax.spec.molinfo import molmass_major_isotope

def test_molmass():
    assert molmass_major_isotope("CO") == pytest.approx(28.01044518292034)

if __name__ == "__main__":
    test_molmass()
    #test_read_HITRAN_molparam()