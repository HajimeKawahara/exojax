from exojax.utils.zsol import nsol
from exojax.utils.zsol import mass_fraction
from exojax.utils.zsol import mass_fraction_XYZ
import numpy as np
import pytest


def test_check_sum_nsol():
    n = nsol()
    sum_n = sum([n[atom] for atom in n])
    assert sum_n == pytest.approx(1.0)


def test_mass_fraction_AAG21():
    n = nsol("AAG21")
    X = mass_fraction("H", n)
    Y = mass_fraction("He", n)
    C = mass_fraction("C", n)

    ref = np.array([0.7438051457070488, 0.24230752749452047, 0.0025561881514610443])
    assert np.allclose([X, Y, C], ref)


def test_solar_abundance_AAG21():
    n = nsol("AAG21")
    X, Y, Z = mass_fraction_XYZ(n)
    ref = np.array([0.7438051457070488, 0.24230752749452047, 0.013887326798430723])

    assert np.allclose([X, Y, Z], ref)


def test_nsol_from_others_AG89():
    n = nsol("AG89")
    X, Y, Z = mass_fraction_XYZ(n)
    ref = np.array([0.7065223726926153, 0.2741121020257724, 0.019365525281612284])

    assert np.allclose([X, Y, Z], ref)


def test_nsol_no_existence_database():
    with pytest.raises(ValueError):
        n = nsol("no_existence_database")


if __name__ == "__main__":
    test_nsol_from_others_AG89()
    #n = nsol("no_existence_database")
    