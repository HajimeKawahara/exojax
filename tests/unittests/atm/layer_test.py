import pytest
import numpy as np
from exojax.atm.atmprof import pressure_layer_logspace

def test_log_pressure_is_constant():
    pressure, dParr, k = pressure_layer_logspace(log_pressure_top=-8.,
                            log_pressure_btm=2.,
                            NP=20,
                            mode='ascending',
                            numpy=False)
    ref_value = 0.70236486
    print(np.log(pressure))
    print(np.log(pressure[1:])-np.log(pressure[:-1]))
        delta_lnP = 1.2118864    
    print(np.exp(1.2118864))
    print(pressure[0], (1-k)*pressure[1])


    assert np.all(dParr/pressure == pytest.approx(np.ones_like(pressure)*ref_value))
    assert 1.0 - k == pytest.approx(ref_value) 


if __name__ == "__main__":
    test_log_pressure_is_constant()