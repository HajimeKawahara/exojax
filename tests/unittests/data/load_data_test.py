#tests for loading exojax data files

def test_read_liquid_ammonia_density():
    from exojax.atm.condensate import read_liquid_ammonia_density
    read_liquid_ammonia_density()

def test_read_mnlist():
    from exojax.utils.isodata import read_mnlist
    read_mnlist()

