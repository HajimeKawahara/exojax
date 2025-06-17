"""Unit tests for spectral band utilities."""

import pytest
import numpy as np
from exojax.utils.spectral_bands import spectral_bands


def test_spectral_bands_linear():
    """Test linear spacing in linear space."""
    nu_bands = spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="linear")
    
    # Check ascending order
    assert np.all(np.diff(nu_bands) > 0)
    
    # Check uniform spacing in linear space
    if len(nu_bands) > 1:
        spacing = np.diff(nu_bands)
        assert np.allclose(spacing, spacing[0], rtol=0.1)


def test_spectral_bands_log():
    """Test logarithmic spacing in log space."""
    nu_bands = spectral_bands(1000.0, 4000.0, band_width=100.0, spacing="log")
    
    # Check ascending order
    assert np.all(np.diff(nu_bands) > 0)
    
    # Check uniform spacing in log space
    if len(nu_bands) > 1:
        log_bands = np.log10(nu_bands)
        log_spacing = np.diff(log_bands)
        assert np.allclose(log_spacing, log_spacing[0], rtol=0.2)


def test_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        spectral_bands(1200.0, 1000.0, band_width=50.0)
    
    with pytest.raises(ValueError):
        spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="invalid")


if __name__ == "__main__":
    test_spectral_bands_linear()
    test_spectral_bands_log()
    test_parameter_validation()
    print("All tests passed!")