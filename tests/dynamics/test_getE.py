import numpy as np
import matplotlib.pyplot as plt
from exojax.dynamics.getE import getE

def test_getE():
    refs=np.array([0.0000000e+00,1.6939898e+00,2.8723323e+00,3.9681163e+00,5.3427000e+00,9.4048452e-01,2.3150694e+00,3.4108529e+00,4.5891953e+00,1.7166138e-05])
    marr=np.linspace(0.0,4*np.pi,10)
    ea=getE(marr,0.3)
    assert np.sum((ea-refs)**2)==0.0

if __name__=="__main__":
    test_getE()
