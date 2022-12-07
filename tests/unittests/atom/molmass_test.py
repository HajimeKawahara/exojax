"""test for molmass calculation"""

import pytest
from exojax.spec.molinfo import molmass


def test_molmass():
    assert molmass('air')==28.97
    assert molmass('CO2')==44.0095
    assert molmass('He')==4.002602
    assert molmass('CO2',db_HIT=True)==44.00974325129166
    assert molmass('He',db_HIT=True)==4.002602


if __name__ == '__main__':
    test_molmass()
