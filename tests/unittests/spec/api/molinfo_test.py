import pytest
from exojax.spec.molinfo import mean_molmass
from exojax.utils.isotopes import molmass_hitran


def test_molmass():
    assert mean_molmass("CO") == pytest.approx(28.01044518292034)

if __name__ == "__main__":
    test_molmass()
    #test_read_HITRAN_molparam()