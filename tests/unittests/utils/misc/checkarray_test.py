"""Tests for array validation utilities."""

import numpy as np
import pytest

from exojax.utils.checkarray import require_ndim


def test_require_ndim_accepts_requested_dimension():
    require_ndim("matrix", np.ones((2, 3)), 2)


def test_require_ndim_reports_name_and_shape():
    with pytest.raises(ValueError, match=r"vector must be 1-dimensional.*\(2, 3\)"):
        require_ndim("vector", np.ones((2, 3)), 1)
