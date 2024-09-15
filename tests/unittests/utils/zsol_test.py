from exojax.utils.zsol import nsol
import pytest

def test_check_sum_nsol():
    n = nsol()
    sum_n = sum([n[atom] for atom in n])
    assert sum_n == pytest.approx(1.0)

from exojax.spec.molinfo import element_mass 

def mmr_metal_solar():
    n = nsol()
    hhe = element_mass["H"]*n["H"] + element_mass["He"]*n["He"]
    sum_element = sum([element_mass[atom]*n[atom] for atom in n])
    mmr_metal = (sum_element - hhe)/sum_element
    
    
    print(element_mass["H"]*n["H"]/sum_element)
    print(element_mass["He"]*n["He"]/sum_element)
    print(mmr_metal)

if __name__ == "__main__":
    test_check_sum_nsol()
    mmr_metal_solar()