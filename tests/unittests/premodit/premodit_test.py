import pytest
from exojax.spec.premodit import compute_dElower

def test_compute_dElower():
    assert compute_dElower(1000.0,interval_contrast=0.1)==pytest.approx(160.03762408883165)
