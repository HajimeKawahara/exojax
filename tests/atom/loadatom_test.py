""" test for loading atomic data
    


"""

import pytest
from exojax.spec.atomllapi import load_atomicdata

def test_loadatom():
    data=load_atomicdata()
    assert data["ionizationE1"][0]==13.595

def test_
    
if __name__ == "__main__":
    test_loadatom()
