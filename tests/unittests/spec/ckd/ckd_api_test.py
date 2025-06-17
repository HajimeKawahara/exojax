"""Unit tests for CKD API grid setting."""

import pytest
import numpy as np
from exojax.opacity.ckd.api import OpaCKD


class MockBaseOpa:
    """Mock base opacity calculator for testing."""
    def __init__(self):
        self.nu_grid = np.linspace(1000.0, 2000.0, 1000)


def test_opa_ckd_init():
    """Test basic OpaCKD initialization."""
    mock_base_opa = MockBaseOpa()
    
    # Test initialization
    opa_ckd = OpaCKD(mock_base_opa, Ng=16, band_width=100.0)
    
    # Check basic attributes
    assert opa_ckd.method == "ckd"
    assert opa_ckd.Ng == 16
    assert opa_ckd.band_spacing == "log"  # Default
    assert len(opa_ckd.nu_bands) > 0
    assert opa_ckd.ready == False


def test_opa_ckd_custom_bands():
    """Test OpaCKD with custom bands."""
    mock_base_opa = MockBaseOpa()
    custom_bands = np.array([1100.0, 1300.0, 1500.0])
    opa_ckd = OpaCKD(mock_base_opa, nu_bands=custom_bands)
    
    np.testing.assert_array_equal(opa_ckd.nu_bands, custom_bands)


if __name__ == "__main__":
    test_opa_ckd_init()
    test_opa_ckd_custom_bands()
    print("All tests passed!")