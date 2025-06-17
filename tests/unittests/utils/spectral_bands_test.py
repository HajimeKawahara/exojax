"""Unit tests for spectral band utilities."""

import pytest
import numpy as np
from exojax.utils.spectral_bands import spectral_bands, spectral_band_edges


def test_spectral_bands_linear():
    """Test linear spacing in linear space."""
    nu_bands, edges = spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="linear")
    
    # Check ascending order
    assert np.all(np.diff(nu_bands) > 0)
    
    # Check uniform spacing in linear space
    if len(nu_bands) > 1:
        spacing = np.diff(nu_bands)
        assert np.allclose(spacing, spacing[0], rtol=0.1)
    
    # Check that edges are returned
    assert edges.shape == (len(nu_bands), 2)


def test_spectral_bands_log():
    """Test logarithmic spacing in log space."""
    nu_bands, edges = spectral_bands(1000.0, 4000.0, band_width=100.0, spacing="log")
    
    # Check ascending order
    assert np.all(np.diff(nu_bands) > 0)
    
    # Check uniform spacing in log space
    if len(nu_bands) > 1:
        log_bands = np.log10(nu_bands)
        log_spacing = np.diff(log_bands)
        assert np.allclose(log_spacing, log_spacing[0], rtol=0.2)
    
    # Check that edges are returned
    assert edges.shape == (len(nu_bands), 2)


def test_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        spectral_bands(1200.0, 1000.0, band_width=50.0)
    
    with pytest.raises(ValueError):
        spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="invalid")




def test_edges_first_consistency():
    """Test that edges-first approach gives consistent results."""
    # Linear spacing test
    nu_min, nu_max = 1000.0, 1200.0
    band_width = 50.0
    
    # Get edges directly
    edges = spectral_band_edges(nu_min, nu_max, band_width, spacing="linear")
    
    # Get centers from edges-first approach
    nu_bands, _ = spectral_bands(nu_min, nu_max, band_width, spacing="linear")
    
    # Verify centers are arithmetic means of edges
    expected_centers = (edges[:, 0] + edges[:, 1]) / 2.0
    np.testing.assert_allclose(nu_bands, expected_centers)
    
    # Log spacing test  
    nu_min, nu_max = 1000.0, 2000.0
    
    # Get edges directly
    edges_log = spectral_band_edges(nu_min, nu_max, band_width, spacing="log")
    
    # Get centers from edges-first approach
    nu_bands_log, _ = spectral_bands(nu_min, nu_max, band_width, spacing="log")
    
    # Verify centers are geometric means of edges
    expected_centers_log = np.sqrt(edges_log[:, 0] * edges_log[:, 1])
    np.testing.assert_allclose(nu_bands_log, expected_centers_log)
    
    # Test that both functions produce consistent results
    assert len(nu_bands_log) == len(edges_log)


def test_edges_primary_linear():
    """Test primary edge calculation for linear spacing."""
    edges = spectral_band_edges(1000.0, 1200.0, 50.0, spacing="linear")
    
    # Check expected edges for linear case
    expected = np.array([[1000, 1050], [1050, 1100], [1100, 1150], [1150, 1200]])
    np.testing.assert_allclose(edges, expected)
    
    # Check edges meet exactly
    for i in range(len(edges) - 1):
        assert edges[i, 1] == edges[i + 1, 0]


def test_edges_primary_log():
    """Test primary edge calculation for log spacing."""
    edges = spectral_band_edges(1000.0, 2000.0, 50.0, spacing="log")
    
    # Check that edges are in ascending order
    assert np.all(edges[:, 0] < edges[:, 1])  # left < right
    assert np.all(edges[:-1, 1] <= edges[1:, 0])  # edges don't overlap
    
    # Check coverage
    assert edges[0, 0] <= 1000.0
    assert edges[-1, 1] >= 2000.0
    
    # Check equal spacing in log space
    # Get unique edge positions (since adjacent bands share edges)
    unique_edges = np.unique(edges.flatten())
    log_edges = np.log10(unique_edges)
    
    # Check uniform spacing in log space
    if len(log_edges) > 1:
        log_spacing = np.diff(log_edges)
        # Allow some tolerance for numerical precision
        assert np.allclose(log_spacing, log_spacing[0], rtol=1e-10)


if __name__ == "__main__":
    test_spectral_bands_linear()
    test_spectral_bands_log()
    test_parameter_validation()
    test_edges_first_consistency()
    test_edges_primary_linear()
    test_edges_primary_log()
    print("All tests passed!")

    exit()
    sp="log"
    #sp="linear"
    nu_bands, edges = spectral_bands(1000.0, 2000.0, band_width=50.0, spacing=sp)
    import matplotlib.pyplot as plt
    plt.plot(nu_bands, np.ones_like(nu_bands), ".")
    plt.plot(edges[:,0], np.ones_like(edges[:,0]), "+")
    plt.plot(edges[:,1], np.ones_like(edges[:,1]), "+")
    plt.xscale("log")
    plt.show()
    exit()

