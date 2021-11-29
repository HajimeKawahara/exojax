""" test for hjert
    
   - This test compares hjert with scipy wofz, see Appendix in Paper I.


"""

import pytest
from exojax.spec.atomllapi import load_atomicdata
import numpy as np

def test_load():
    data=load_atomicdata()
    print(data)
#    assert np.max(diffarr)<1.e-6

if __name__ == "__main__":
    test_load()
