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


def compute_ckd_single_tp(
    T: float,
    P: float,
    base_opa,
    g_grid: jnp.ndarray
) -> jnp.ndarray:
    """Compute CKD log_kggrid for single T,P condition.
    
    Pure JAX function that computes the complete CKD workflow
    for a single temperature and pressure condition.
    
    Args:
        T: Temperature in Kelvin
        P: Pressure in bar
        base_opa: Base opacity calculator (OpaPremodit, OpaModit, etc.)
        g_grid: Gauss-Legendre g-ordinates, shape (Ng,)
        
    Returns:
        log_kggrid: Log k-values on g-grid, shape (Ng,)
    """
    # Step 1: Compute cross-section vector
    xsv = base_opa.xsvector(T, P)
    
    # Step 2: Compute g-ordinates and sorted k-values
    idx, k_g, g = compute_g_ordinates(xsv)
    
    # Step 3: Safe logarithm
    log_k_g = safe_log_k(k_g)
    
    # Step 4: Interpolate to g-grid
    log_kggrid = interpolate_log_k_to_g_grid(g, log_k_g, g_grid)
    
    return log_kggrid


def compute_ckd_tp_grid_vectorized(
    T_grid: Union[np.ndarray, jnp.ndarray],
    P_grid: Union[np.ndarray, jnp.ndarray], 
    base_opa,
    Ng: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute CKD tables for temperature and pressure grids using vmap.
    
    Fully vectorized implementation that processes all T,P combinations
    simultaneously without loops.
    
    Args:
        T_grid: Temperature grid in Kelvin, shape (nT,)
        P_grid: Pressure grid in bar, shape (nP,)
        base_opa: Base opacity calculator instance
        Ng: Number of Gauss-Legendre quadrature points
        
    Returns:
        Tuple of:
            - log_kggrid: CKD tables, shape (nT, nP, Ng)
            - ggrid: G-ordinates, shape (Ng,)
            - weights: Quadrature weights, shape (Ng,)
    """
    T_grid = jnp.asarray(T_grid)
    P_grid = jnp.asarray(P_grid)
    nT, nP = len(T_grid), len(P_grid)
    
    # Generate Gauss-Legendre grid once
    ggrid, weights = gauss_legendre_grid(Ng)
    
    # Create meshgrid and flatten for xsmatrix
    T_mesh, P_mesh = jnp.meshgrid(T_grid, P_grid, indexing='ij')
    T_flat = T_mesh.flatten()  # Shape: (nT*nP,)
    P_flat = P_mesh.flatten()  # Shape: (nT*nP,)
    
    # Compute cross-section matrix for all flattened T,P combinations
    # xsmatrix shape: (nT*nP, nnu)
    xsmatrix_flat = base_opa.xsmatrix(T_flat, P_flat)
    
    # Reshape back to grid format: (nT*nP, nnu) -> (nT, nP, nnu)
    nnu = xsmatrix_flat.shape[1]
    xsmatrix = xsmatrix_flat.reshape(nT, nP, nnu)
    
    def process_single_spectrum(xsv: jnp.ndarray) -> jnp.ndarray:
        """Process single cross-section vector to log_kggrid."""
        # Compute g-ordinates and sorted k-values
        idx, k_g, g = compute_g_ordinates(xsv)
        
        # Safe logarithm
        log_k_g = safe_log_k(k_g)
        
        # Interpolate to g-grid
        log_kg_single = interpolate_log_k_to_g_grid(g, log_k_g, ggrid)
        
        return log_kg_single
    
    # Vectorize properly: we need to process each (i,j) spectrum separately
    # Input xsmatrix has shape (nT, nP, nnu)
    # We want output shape (nT, nP, Ng)
    
    # Method 1: Flatten and reshape approach
    # Flatten to (nT*nP, nnu), process, then reshape to (nT, nP, Ng)
    xsmatrix_flat_for_processing = xsmatrix.reshape(nT * nP, nnu)
    
    # Vectorize over the flattened batch dimension
    process_batch = vmap(process_single_spectrum, in_axes=0, out_axes=0)
    log_kggrid_flat = process_batch(xsmatrix_flat_for_processing)
    
    # Reshape back to (nT, nP, Ng)
    log_kggrid = log_kggrid_flat.reshape(nT, nP, Ng)
    
    return log_kggrid, ggrid, weights

