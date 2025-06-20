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
    
    # Check that band_edges are now available (new feature)
    assert hasattr(opa_ckd, 'band_edges')
    assert opa_ckd.band_edges.shape == (len(opa_ckd.nu_bands), 2)


def test_opa_ckd_custom_bands():
    """Test OpaCKD with custom band settings."""
    mock_base_opa = MockBaseOpa()
    
    # Test with different band width and spacing
    opa_ckd = OpaCKD(mock_base_opa, band_width=200.0, band_spacing="linear")
    
    # Check that bands were auto-generated with the specified parameters
    assert opa_ckd.band_width == 200.0
    assert opa_ckd.band_spacing == "linear"
    assert len(opa_ckd.nu_bands) > 0
    assert hasattr(opa_ckd, 'band_edges')


if __name__ == "__main__":
    test_opa_ckd_init()
    test_opa_ckd_custom_bands()
    print("All tests passed!")