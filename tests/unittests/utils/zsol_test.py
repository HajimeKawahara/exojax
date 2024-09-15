from exojax.utils.zsol import nsol
import pytest

def test_check_sum_nsol():
    n = nsol()
    sum_n = sum([n[atom] for atom in n])
    assert sum_n == pytest.approx(1.0)