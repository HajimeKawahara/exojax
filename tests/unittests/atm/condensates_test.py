from exojax.atm.condensate import condensate_substance_density
from exojax.atm.condensate import condensate_density_liquid_ammonia
    
def test_condensate_density():
    assert (condensate_substance_density["Fe"]) == 7.875
    
def test_condensate_density_ammonia():
    assert (condensate_density_liquid_ammonia(150.0)) == 0.73094

if __name__ == "__main__":
    test_condensate_density()