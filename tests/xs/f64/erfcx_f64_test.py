""" test for erfcx
    
   - This test compares hjert with scipy.erfcx, see Appendix in Paper I.


"""

import pytest
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.special import erfcx as sc_erfcx
from exojax.special import erfcx
from exojax.special._special import erfcx_scan
from jax.config import config                                                  
config.update("jax_enable_x64", True)

N=10000
xv=jnp.logspace(-5,5,N)
xvc=np.logspace(-5,5,N)
verfcx=vmap(erfcx)
verfcx_scan=vmap(erfcx_scan)
ref=sc_erfcx(xvc)

def test_comparison_erfcx_scipy():
    d=(verfcx(xv) - ref)/ ref
    assert np.max(d)<2.e-6

def test_comparison_erfcx_scan_scipy():
    d=(verfcx_scan(xv) - ref)/ ref
    assert np.max(d)<2.e-6

    
if __name__ == "__main__":
    import time
    test_comparison_erfcx_scipy()
    test_comparison_erfcx_scan_scipy()

    if False:
    #comparison 
        for j in range(0,3):
            ts=time.time()
            for i in range(0,10000):
                verfcx(xv)
            te=time.time()
            print("direct",te-ts)
            
            ts=time.time()
            for i in range(0,10000):
                verfcx_scan(xv)
            te=time.time()
            print("scan",te-ts)
