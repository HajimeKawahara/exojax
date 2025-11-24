from exojax.postproc.limb_darkening import ld_kipping
import pytest

def test_ld_kipping():
    u1,u2=ld_kipping(0.5,0.5)
    assert u1 == pytest.approx(0.70710677)
    assert u2 == pytest.approx(0.0)
    
if __name__ == "__main__":
    test_ld_kipping()