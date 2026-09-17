import numpy as np
import pytest

from exojax.utils.checkarray import is_outside_range, is_sorted


def test_is_sorted():
    import numpy as np

    a = np.array([1, 2, 3])
    b = np.array([1, 3, 2])

    assert is_sorted(a) == "ascending"
    assert is_sorted(a[::-1]) == "descending"
    assert is_sorted(b) == "unordered"
    assert is_sorted(2.0) == "single"


@pytest.mark.parametrize(
    "xarr, xs, xe, expected",
    [
        (np.array([1.2, 1.4, 1.7, 1.3, 1.0]), 0.7, 0.8, True),  # No element in range
        (np.array([0.75, 1.0, 1.5]), 0.7, 0.8, False),  # One element in range
        (np.array([0.6, 0.9, 1.2]), 0.7, 0.8, True),  # No element in range
        (np.array([]), 0.5, 1.0, True),  # Empty array
        (np.array([0.5, 1.0]), 0.5, 1.0, True),  # Boundary values
    ],
)
def test_is_outside_range(xarr, xs, xe, expected):
    assert is_outside_range(xarr, xs, xe) == expected
