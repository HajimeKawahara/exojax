from exojax.utils.isotopes import get_isotope
from exojax.utils.isotopes import get_stable_isotope
from exojax.utils.isotopes import isodata
import numpy as np


def test_get_isotope():
    isolist = isodata.read_mnlist()
    ref = (['1H', '2H', '3H'], [1.007825, 2.014102,
                                3.016049], [99.9885, 0.0115, np.nan])
    assert np.all(get_isotope('H', isolist)[0:2] == ref[0:2])
    assert (get_isotope('H', isolist)[2][2] != get_isotope('H', isolist)[2][2])
    assert np.all(np.array(get_isotope('H', isolist)[2][0:2]) == ref[2][0:2])

def test_get_stable_isotope():
    isolist = isodata.read_mnlist()
    ref = ('1H', 1.007825, 99.9885)
    assert np.all(get_stable_isotope('H', isolist) == ref)


if __name__ == "__main__":
    test_get_isotope()
    test_get_stable_isotope()