import pytest
from exojax.special import j0
from scipy.special import j0 as j0_scipy

def test_j0():
    x=1.0
    assert j0(x) == pytest.approx(j0_scipy(x))
    
if __name__ == "__main__":
    test_j0()