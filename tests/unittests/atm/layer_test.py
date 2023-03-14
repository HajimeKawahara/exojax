import pytest
import numpy as np
from exojax.atm.atmprof import pressure_layer_logspace

def test_log_pressure_is_constant():
    pressure, dParr, k = pressure_layer_logspace(log_pressure_top=-8.,
                            log_pressure_btm=2.,
                            NP=20,
                            mode='ascending',
                            numpy=False)
    
    #check P[n-1] = k P[n]
    assert np.all(np.abs(1.0 - pressure[1:]*k/pressure[:-1]) < 1.e-5)
    #check dParr
    assert np.all(np.abs(1.0-(pressure[1:] - pressure[:-1])/dParr[1:]) < 1.e-5)
    
    #assert np.all(dParr/pressure == pytest.approx(np.ones_like(pressure)*ref_value))
    #assert 1.0 - k == pytest.approx(ref_value) 


if __name__ == "__main__":
    test_log_pressure_is_constant()