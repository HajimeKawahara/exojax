import pytest
from exojax.spec.molinfo import molmass_isotope
from exojax.spec.molinfo import isotope_molmass


def test_isotope_molmass():
    assert isotope_molmass("12C-16O") == pytest.approx(27.994915)

def test_mean_molmass():
    assert molmass_isotope("CO") == pytest.approx(28.01044518292034)

if __name__ == "__main__":
    test_isotope_molmass()
    test_mean_molmass()
    #test_read_HITRAN_molparam()