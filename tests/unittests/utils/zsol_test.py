from exojax.utils.zsol import nsol
from exojax.utils.zsol import mass_fraction_XYZ
import pytest

def test_check_sum_nsol():
    n = nsol()
    sum_n = sum([n[atom] for atom in n])
    assert sum_n == pytest.approx(1.0)

def test_solar_abundance():
    n = nsol()
    X, Y, Z = mass_fraction_XYZ(n)
    #ref = np.array([])
    
    print(X, Y, Z)

if __name__ == "__main__":
    test_check_sum_nsol()
    test_solar_abundance()