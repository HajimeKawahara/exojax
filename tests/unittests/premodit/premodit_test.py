import pytest
from exojax.spec.premodit import compute_dElower, make_elower_grid

def test_compute_dElower():
    assert compute_dElower(1000.0,interval_contrast=0.1)==pytest.approx(160.03762408883165)

def test_make_elower_grid():
    maxe=12001.0
    mine=99.01
    eg=make_elower_grid(1000, [mine,maxe], 1.0)
    assert eg[-1]>=maxe and eg[0]<=mine
