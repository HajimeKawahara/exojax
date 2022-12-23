"""test for molmass calculation"""

import pytest
from exojax.spec.molinfo import mean_molmass


def test_molmass():
    assert mean_molmass('air')==28.97
    assert mean_molmass('CO2')==44.0095
    assert mean_molmass('He')==4.002602
    assert mean_molmass('CO2',db_HIT=True)==44.00974325129166
    assert mean_molmass('He',db_HIT=True)==4.002602


if __name__ == '__main__':
    test_molmass()
