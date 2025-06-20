"""Unit tests for CKD core algorithms."""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import config

from exojax.opacity.ckd.core import (
    compute_g_ordinates, 
    gauss_legendre_grid,
    safe_log_k,
    interpolate_log_k_to_g_grid,
    compute_ckd_from_xsv,
    compute_ckd_tables
)

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


def test_safe_log_k():
    """Test safe_log_k handles zeros correctly and uses precision-aware defaults."""
    k_values = jnp.array([1.0, 0.0, 2.0])
    
    # Test with explicit min_value
    log_k = safe_log_k(k_values, 1e-30)
    assert jnp.isclose(log_k[0], 0.0)  # log(1) = 0
    assert jnp.isfinite(log_k[1])      # log(0) replaced with finite value
    assert jnp.isclose(log_k[2], jnp.log(2.0))  # log(2)
    
    # Test precision-aware defaults
    k_values_f32 = jnp.array([1.0, 0.0, 2.0], dtype=jnp.float32)
    k_values_f64 = jnp.array([1.0, 0.0, 2.0], dtype=jnp.float64)
    
    log_k_f32 = safe_log_k(k_values_f32)  # Should use 1e-30
    log_k_f64 = safe_log_k(k_values_f64)  # Should use 1e-100
    
    # Both should be finite but f64 should allow smaller values
    assert jnp.isfinite(log_k_f32[1])
    assert jnp.isfinite(log_k_f64[1])
    assert log_k_f64[1] < log_k_f32[1]  # 1e-100 gives smaller log than 1e-30


def test_interpolate_log_k_to_g_grid():
    """Test interpolation matches jnp.interp behavior."""
    g_ordinates = jnp.array([0.0, 0.5, 1.0])
    log_k_sorted = jnp.array([1.0, 2.0, 3.0])
    g_grid = jnp.array([0.0, 0.25, 0.5, 1.0])
    
    result = interpolate_log_k_to_g_grid(g_ordinates, log_k_sorted, g_grid)
    expected = jnp.interp(g_grid, g_ordinates, log_k_sorted)
    
    assert jnp.allclose(result, expected)


def test_compute_ckd_tables():
    """Test CKD table computation produces correct shapes."""
    from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
    from exojax.opacity.opacalc import OpaPremodit
    
    # Small test case
    nus, wav, res = mock_wavenumber_grid(lambda0=22930.0, lambda1=22932.0, Nx=1000)
    mdb = mock_mdbExomol("H2O") 
    opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])
    
    T_grid = jnp.array([800.0, 1000.0])
    P_grid = jnp.array([0.1, 1.0])
    Ng = 8
    
    # Create T,P meshgrid and compute cross-section matrix
    T_mesh, P_mesh = jnp.meshgrid(T_grid, P_grid, indexing='ij')
    T_flat = T_mesh.flatten()
    P_flat = P_mesh.flatten()
    xsmatrix_flat = opa.xsmatrix(T_flat, P_flat)
    nnu = xsmatrix_flat.shape[1]
    xsmatrix = xsmatrix_flat.reshape(2, 2, nnu)
    
    log_kggrid, ggrid, weights = compute_ckd_tables(xsmatrix, Ng)
    
    # Check shapes
    assert log_kggrid.shape == (2, 2, 8)  # (nT, nP, Ng)
    assert ggrid.shape == (8,)
    assert weights.shape == (8,)
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(log_kggrid))
    assert jnp.all(jnp.isfinite(ggrid))
    assert jnp.all(jnp.isfinite(weights))


if __name__ == "__main__":
    test_compute_g_ordinates_basic()
    test_compute_g_ordinates_properties()
    test_gauss_legendre_grid_basic()
    test_gauss_legendre_grid_transformation()
    test_safe_log_k()
    test_interpolate_log_k_to_g_grid()
    test_compute_ckd_tables()
    print("All tests passed!")