import jax.numpy as jnp
import pytest
import numpy as np
from exojax.utils.indexing import getix, npgetix

def test_getix_ascending():
    x=jnp.array([0.7,1.3])
    xv=jnp.array([0.0,1.0,2.0])
    c,i=getix(x, xv)
    assert c[0]==pytest.approx(0.7)
    assert c[1]==pytest.approx(0.3)
    assert i[0]==0
    assert i[1]==1


def test_npgetix_ascending():
    x=np.array([0.7,1.3])
    xv=np.array([0.0,1.0,2.0])
    c,i=npgetix(x, xv)
    assert c[0]==pytest.approx(0.7)
    assert c[1]==pytest.approx(0.3)
    assert i[0]==0
    assert i[1]==1

def test_npgetix_descending():
    x=np.array([0.7,3.3])
    xv=np.array([4.0,3.0,2.0,1.0,0.0])
    c,i=npgetix(x, xv)
    print(c)
    print(xv[i])


if __name__=="__main__":
    test_npgetix_ascending()
    #test_npgetix_descending()
#    test_getix_ascending()