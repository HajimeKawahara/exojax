import pytest
import numpy as np
from exojax.spec.setrt import gen_wavenumber_grid

def test_gen_wavenumber_grid():
    Nx=4000
    nus, wav, res = gen_wavenumber_grid(29200.0,29300., Nx, unit='AA')
    dif=np.log(nus[1:])-np.log(nus[:-1])
    refval=8.54915417e-07
    assert np.all(dif==pytest.approx(refval*np.ones_like(dif)))
    
if __name__=="__main__":
    test_gen_wavenumber_grid()
