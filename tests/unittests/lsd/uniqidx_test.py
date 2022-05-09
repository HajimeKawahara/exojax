import numpy as np
from exojax.spec.lsd import uniqidx, uniqidx_2D

def test_uniqidx():
    a=np.array([4,7,7,7,8,4])
    uidx=uniqidx(a)
    ref=np.array([0,1,1,1,2,0])
    diff=uidx-ref
    assert np.all(diff==0.0)

def test_uniqidx_2D():
    a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])
    uidx=uniqidx_2D(a)
    ref=np.array([0,1,2,1,3,0])
    diff=uidx-ref
    assert np.all(diff==0.0)

    
