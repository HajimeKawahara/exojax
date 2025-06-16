"""Unit tests for CKD core algorithms."""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import config

from exojax.opacity.ckd.core import compute_g_ordinates, gauss_legendre_grid

config.update("jax_enable_x64", True)


def test_compute_g_ordinates_basic():
    """Test basic functionality of compute_g_ordinates."""
    # Simple test case with known values
    xsv = jnp.array([3.0, 1.0, 4.0, 2.0])
    
    idx, k_g, g = compute_g_ordinates(xsv)
    
    # Check sorting indices
    expected_idx = jnp.array([1, 3, 0, 2])  # indices that sort [3,1,4,2] -> [1,2,3,4]
    assert jnp.allclose(idx, expected_idx)
    
    # Check sorted k-values
    expected_k_g = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(k_g, expected_k_g)
    
    # Check g-ordinates
    expected_g = jnp.array([0.0, 0.25, 0.5, 0.75])
    assert jnp.allclose(g, expected_g)


def test_compute_g_ordinates_properties():
    """Test key mathematical properties."""
    np.random.seed(42)
    xsv = jnp.array(np.random.lognormal(0, 2, 100))
    
    idx, k_g, g = compute_g_ordinates(xsv)
    
    # k_g should be sorted in ascending order
    assert jnp.all(k_g[1:] >= k_g[:-1])
    
    # g should be uniform distribution from 0 to 1
    expected_g = jnp.arange(len(xsv)) / len(xsv)
    assert jnp.allclose(g, expected_g)
    
    # Indices should correctly map original to sorted
    assert jnp.allclose(xsv[idx], k_g)


def test_gauss_legendre_grid_basic():
    """Test basic functionality of gauss_legendre_grid."""
    Ng = 4
    gpoints, weights = gauss_legendre_grid(Ng)
    
    # Check array lengths
    assert len(gpoints) == Ng
    assert len(weights) == Ng
    
    # Check interval [0,1]
    assert jnp.all(gpoints >= 0.0) and jnp.all(gpoints <= 1.0)
    
    # Check ordering (should be ascending)
    assert jnp.all(gpoints[1:] >= gpoints[:-1])
    
    # Check weight properties (should be positive)
    assert jnp.all(weights > 0.0)


def test_gauss_legendre_grid_transformation():
    """Test the [-1,1] to [0,1] transformation logic."""
    Ng = 8
    gpoints, weights = gauss_legendre_grid(Ng)
    
    # Get original [-1,1] points for comparison
    x_orig, w_orig = np.polynomial.legendre.leggauss(Ng)
    x_orig, w_orig = jnp.array(x_orig), jnp.array(w_orig)
    
    # Check our transformation: gpoints = 0.5 * (1.0 + x_orig)
    expected_gpoints = 0.5 * (1.0 + x_orig)
    assert jnp.allclose(gpoints, expected_gpoints)
    
    # Check our transformation: weights = 0.5 * w_orig  
    expected_weights = 0.5 * w_orig
    assert jnp.allclose(weights, expected_weights)


if __name__ == "__main__":
    test_compute_g_ordinates_basic()
    test_compute_g_ordinates_properties()
    test_gauss_legendre_grid_basic()
    test_gauss_legendre_grid_transformation()
    print("All tests passed!")