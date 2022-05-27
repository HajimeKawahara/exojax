import numpy as np
from exojax.spec.lsd import uniqidx, uniqidx_2D

def test_uniqidx():
    a=np.array([4,7,7,7,8,4])
    uidx,val=uniqidx(a)
    ref=np.array([0,1,1,1,2,0])
    assert np.all(uidx-ref==0.0)
    refval=np.array([4,7,8])    
    assert np.all(val-refval==0.0)
    
def test_uniqidx_2D():
    a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])
    uidx,val=uniqidx_2D(a)
    ref=np.array([0,1,2,1,3,0])
    assert np.all(uidx-ref==0.0)
    refval=np.array([[4,1],[7,1],[7,2],[8,0]])
    assert np.all(val-refval==0.0)
    
if __name__=="__main__":
    test_uniqidx()
    test_uniqidx_2D()
