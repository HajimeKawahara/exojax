import pytest
from exojax.utils.moltex import format_molecule

def test_format_molecule():
    assert format_molecule("H2O") == "$\\mathrm{H_2O}$"
    assert format_molecule("CH4") == "$\\mathrm{CH_4}$"
    assert format_molecule("CO") == "$\\mathrm{CO}$"
    assert format_molecule("NH3") == "$\\mathrm{NH_3}$"