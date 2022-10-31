from exojax.spec.api import _convert_proper_isotope
from exojax.spec.api import _isotope_index_from_isotope_number
import numpy as np


def test__convert_proper_isotope():
    assert _convert_proper_isotope(0) is None
    assert _convert_proper_isotope(1) == "1"
    assert _convert_proper_isotope(None) is None


def test__isotope_index_from_isotope_number():
    uniqiso = np.array([1, 2, 3, 5, 6])
    assert _isotope_index_from_isotope_number(1, uniqiso) == 0
    assert _isotope_index_from_isotope_number(6, uniqiso) == 4


if __name__ == "__main__":
    test__convert_proper_isotope()
    test__isotope_index_from_isotope_number()