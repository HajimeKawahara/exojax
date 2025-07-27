"""Core algorithms for Correlated-K Distribution (CKD) calculations.

This module contains pure JAX functions for CKD computations including
g-ordinate calculations, k-distribution sorting, and quadrature operations.
All functions are designed to be JAX-transformable (jit, vmap, grad).
"""

from __future__ import annotations
from functools import partial
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


@jit
def compute_g_ordinates(xsv: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute g-ordinates from cross-section vector.
    
    Pure JAX function that sorts cross-sections and computes cumulative
    probability distribution (g-ordinates) for k-distribution method.
    
    The g-ordinates represent the cumulative probability distribution where:
    - g=0 corresponds to the smallest k-value (most transparent)
    - g=1 corresponds to the largest k-value (most opaque)
    - g-values are uniformly distributed between 0 and 1
    
    Args:
        xsv: Cross-section vector, shape (nnu,)
        
    Returns:
        Tuple of:
            - idx: Sorting indices, shape (nnu,)
            - k_g: Sorted k-values (ascending), shape (nnu,)
            - g: Cumulative probability ordinates [0, 1], shape (nnu,)
    """
    # Sort cross-sections in ascending order (smallest to largest k)
    idx = jnp.argsort(xsv)
    k_g = xsv[idx]
    
    # Compute g-ordinates as uniform distribution from 0 to 1
    # g[i] = i / N, where i goes from 0 to N-1
    n_points = xsv.shape[0]
    g = jnp.arange(n_points, dtype=xsv.dtype) / n_points
    
    return idx, k_g, g


@partial(jit, static_argnums=(0,))
def gauss_legendre_grid(Ng: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Gauss-Legendre quadrature points and weights on [0,1].
    
    Pure JAX function for generating quadrature grid. Should be called
    during initialization, not in runtime hot paths.
    
    Transforms the standard Gauss-Legendre quadrature from [-1,1] to [0,1]
    interval for use in correlated-k distribution integration.
    
    Args:
        Ng: Number of quadrature points (static argument)
        
    Returns:
        Tuple of:
            - gpoints: Quadrature points on [0,1], shape (Ng,)
            - weights: Quadrature weights, shape (Ng,)
    """
    # Generate standard Gauss-Legendre points and weights on [-1,1]
    # Note: Use numpy for this since JAX doesn't have polynomial.legendre.leggauss
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(Ng)
    x, w = jnp.array(x), jnp.array(w)
    
    # Transform from [-1,1] to [0,1] interval
    # For interval transformation: [a,b] -> [c,d]
    # x_new = (d-c)/2 * x_old + (d+c)/2
    # w_new = (d-c)/2 * w_old
    # Here: [-1,1] -> [0,1], so (d-c)/2 = 0.5, (d+c)/2 = 0.5
    gpoints = 0.5 * (1.0 + x)  # Transform points to [0,1]
    weights = 0.5 * w          # Scale weights accordingly
    
    return gpoints, weights


@jit
def safe_log_k(k_values: jnp.ndarray, min_value: float = None) -> jnp.ndarray:
    """Compute safe logarithm of k-values avoiding log(0).
    
    Pure JAX function that computes log(k) while avoiding numerical
    issues from zero or negative k-values.
    
    Args:
        k_values: K-values (cross-sections), shape (nnu,)
        min_value: Minimum value to avoid log(0). If None, uses precision-aware default:
                  1e-100 for float64, 1e-30 for float32
        
    Returns:
        log_k: Safe logarithm of k-values, shape (nnu,)
    """
    if min_value is None:
        if k_values.dtype == jnp.float64:
            min_value = 1e-100  # Much smaller for float64
        else:
            min_value = 1e-30   # Current default for float32
    return jnp.log(jnp.maximum(k_values, min_value))


@jit
def interpolate_log_k_to_g_grid(
    g_ordinates: jnp.ndarray,
    log_k_sorted: jnp.ndarray,
    g_grid: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate log(k) values to Gauss-Legendre g-grid.
    
    Pure JAX function for interpolating sorted k-values onto the
    quadrature grid for efficient radiative transfer integration.
    
    Args:
        g_ordinates: Original g-ordinates from spectrum, shape (nnu,)
        log_k_sorted: Log of sorted k-values, shape (nnu,)
        g_grid: Target Gauss-Legendre g-grid, shape (Ng,)
        
    Returns:
        log_kg_grid: Interpolated log(k) on g-grid, shape (Ng,)
    """
    # Use JAX's interp function for linear interpolation
    # This matches the functionality from corrk_table.py:
    # log_kggrid = jnp.interp(ggrid, g, log_k_g)
    log_kg_grid = jnp.interp(g_grid, g_ordinates, log_k_sorted)
    return log_kg_grid


@jit
def compute_ckd_from_xsv(
    xsv: jnp.ndarray,
    g_grid: jnp.ndarray
) -> jnp.ndarray:
    """Compute CKD log_kggrid from cross-section vector.
    
    Pure JAX function that computes the complete CKD workflow
    from a cross-section vector to log k-values on g-grid.
    
    Args:
        xsv: Cross-section vector, shape (nnu,)
        g_grid: Gauss-Legendre g-ordinates, shape (Ng,)
        
    Returns:
        log_kggrid: Log k-values on g-grid, shape (Ng,)
    """
    # Step 1: Compute g-ordinates and sorted k-values
    idx, k_g, g = compute_g_ordinates(xsv)
    
    # Step 2: Safe logarithm
    log_k_g = safe_log_k(k_g)
    
    # Step 3: Interpolate to g-grid
    log_kggrid = interpolate_log_k_to_g_grid(g, log_k_g, g_grid)
    
    return log_kggrid


@jit
def compute_ckd_from_xsmatrix(
    xsmatrix: jnp.ndarray,
    g_grid: jnp.ndarray
) -> jnp.ndarray:
    """Compute CKD log_kggrid from cross-section matrix.
    
    Pure JAX function that processes a cross-section matrix to compute
    CKD tables using vectorized operations.
    
    Args:
        xsmatrix: Cross-section matrix, shape (nT, nP, nnu) or (batch, nnu)
        g_grid: Gauss-Legendre g-ordinates, shape (Ng,)
        
    Returns:
        log_kggrid: CKD tables, shape matching input batch dimensions + (Ng,)
    """
    original_shape = xsmatrix.shape
    
    # Handle different input shapes
    if len(original_shape) == 2:
        # Shape: (batch, nnu) -> process directly
        batch_size, nnu = original_shape
        xsmatrix_flat = xsmatrix
        output_shape = (batch_size, len(g_grid))
    elif len(original_shape) == 3:
        # Shape: (nT, nP, nnu) -> flatten to (nT*nP, nnu)
        nT, nP, nnu = original_shape
        batch_size = nT * nP
        xsmatrix_flat = xsmatrix.reshape(batch_size, nnu)
        output_shape = (nT, nP, len(g_grid))
    else:
        raise ValueError(f"Unsupported xsmatrix shape: {original_shape}")
    
    # Vectorize the single spectrum processing function
    process_batch = vmap(compute_ckd_from_xsv, in_axes=(0, None), out_axes=0)
    log_kggrid_flat = process_batch(xsmatrix_flat, g_grid)
    
    # Reshape back to original batch structure
    if len(original_shape) == 3:
        log_kggrid = log_kggrid_flat.reshape(output_shape)
    else:
        log_kggrid = log_kggrid_flat
    
    return log_kggrid


def compute_ckd_tables(
    xsmatrix: jnp.ndarray,
    Ng: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute complete CKD tables from pre-computed cross-section matrix.
    
    Pure JAX function that takes pre-computed cross-sections and generates
    complete CKD tables including quadrature grid and weights. The opacity 
    calculation is decoupled from this function.
    
    Args:
        xsmatrix: Pre-computed cross-section matrix, shape (nT, nP, nnu)
        Ng: Number of Gauss-Legendre quadrature points
        
    Returns:
        Tuple of:
            - log_kggrid: CKD tables, shape (nT, nP, Ng)
            - ggrid: G-ordinates, shape (Ng,)
            - weights: Quadrature weights, shape (Ng,)
    """
    # Generate Gauss-Legendre grid
    ggrid, weights = gauss_legendre_grid(Ng)
    
    # Process cross-section matrix to CKD tables
    log_kggrid = compute_ckd_from_xsmatrix(xsmatrix, ggrid)
    
    return log_kggrid, ggrid, weights


@jit  
def interpolate_log_k_2d(
    log_kggrid: jnp.ndarray,
    T_grid: jnp.ndarray,
    P_grid: jnp.ndarray, 
    T: float,
    P: float
) -> jnp.ndarray:
    """JAX-compatible 2D interpolation of log_kggrid at given T,P.
    
    Pure JAX function for interpolating pre-computed CKD tables at a specific
    temperature and pressure point. Uses vectorized operations for efficiency.
    
    Args:
        log_kggrid: Pre-computed log k-values, shape (nT, nP, Ng, nnu_bands)
        T_grid: Temperature grid in Kelvin, shape (nT,)
        P_grid: Pressure grid in bar, shape (nP,)
        T: Target temperature in Kelvin
        P: Target pressure in bar
        
    Returns:
        Interpolated log k-values, shape (Ng, nnu_bands)
    """
    # log_kggrid shape: (nT, nP, Ng, nnu_bands)
    # Vectorized interpolation approach
    
    def interpolate_2d_slice(log_k_2d_slice):
        """Interpolate single 2D slice (nT, nP) at given T,P."""
        # log_k_2d_slice shape: (nT, nP)
        # First interpolate over T dimension for each P
        def interp_over_T(log_k_column):
            """Interpolate over T for single P column."""
            return jnp.interp(T, T_grid, log_k_column)
        
        # Apply to each P column: (nT, nP) -> (nP,)
        log_k_T = vmap(interp_over_T, in_axes=1)(log_k_2d_slice)
        
        # Then interpolate over P dimension in log scale: (nP,) -> scalar
        # Use log scale for pressure due to wide dynamic range
        log_k_TP = jnp.interp(jnp.log(P), jnp.log(P_grid), log_k_T)
        
        return log_k_TP
    
    # Apply vectorized interpolation over (Ng, nnu_bands) dimensions
    # Reshape from (nT, nP, Ng, nnu_bands) to (Ng*nnu_bands, nT, nP)
    nT, nP, Ng, nnu_bands = log_kggrid.shape
    log_k_reshaped = log_kggrid.transpose(2, 3, 0, 1).reshape(-1, nT, nP)
    
    # Vectorize interpolation over all (g, band) combinations
    log_k_flat = vmap(interpolate_2d_slice)(log_k_reshaped)  # Shape: (Ng*nnu_bands,)
    
    # Reshape back to (Ng, nnu_bands)
    log_k_interp = log_k_flat.reshape(Ng, nnu_bands)
    
    return log_k_interp

