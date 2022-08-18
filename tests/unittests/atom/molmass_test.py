"""test for molmass calculation"""

import pytest
from exojax.spec.molinfo import molmass


def test_molmass():
    print(molmass('air'))
    print(molmass('CO2'))
    print(molmass('He'))
    print(molmass('CO2',db_HIT=True))
    print(molmass('He',db_HIT=True))


if __name__ == '__main__':
    test_molmass()
