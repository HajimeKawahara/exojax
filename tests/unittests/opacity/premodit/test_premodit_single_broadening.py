import numpy as np

from exojax.opacity.premodit.premodit import _check_single_broadening


def test_check_single_broadening_with_single_value():
    assert _check_single_broadening(np.array([1.0]), np.array([0.5])) is True
