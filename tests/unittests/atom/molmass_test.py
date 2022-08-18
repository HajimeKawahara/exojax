"""test for molmass calculation"""

import pytest
from exojax.spec.molinfo import molmass


def test_molmass():
    molmass('air')
    molmass('CO2')
    molmass('He')
    molmass('CO2',db_HIT=True)
    molmass('He',db_HIT=True)


if __name__ == '__main__':
    test_molmass()
