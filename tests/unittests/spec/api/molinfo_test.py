import pytest
from exojax.database.molinfo  import molmass_isotope
from exojax.database.molinfo  import isotope_molmass
from exojax.database.molinfo  import molmass #deprecated


def test_isotope_molmass():
    assert isotope_molmass("(12C)(16O)") == pytest.approx(27.994915)
    assert isotope_molmass("12C-16O") == pytest.approx(27.994915)


def test_mean_molmass():
    assert molmass_isotope("CO") == pytest.approx(27.994915)


def test_molmass_CO():
    assert molmass("CO") == pytest.approx(27.994915)


def test_molmass_H2O():
    assert molmass("H2O") == pytest.approx(18.010565)


def test_molmass_CH4():
    assert molmass("CH4") == pytest.approx(16.0313)

