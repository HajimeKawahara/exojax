"""Spectral band and subgrid utilities.

This module provides utilities for generating spectral bands and extracting
subgrids from full wavenumber grids. These functions are useful for:
- Correlated-K Distribution (CKD) computations
"""

import numpy as np


def _set_grid_eslog(x0, x1, N):
    """Generate logarithmically spaced grid (ESLOG)."""
    return np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)


def _set_grid_eslin(x0, x1, N):
    """Generate linearly spaced grid (ESLIN)."""
    return np.linspace(x0, x1, N, dtype=np.float64)


def spectral_band_edges(nu_min, nu_max, band_width=50.0, spacing="linear"):
    """Generate spectral band edges for banded calculations (primary function).
    
    Creates spectral band edges covering the range [nu_min, nu_max] with either
    linear or logarithmic spacing. This edges-first approach ensures precise
    and mathematically consistent band boundaries for CKD computation.
    
    Args:
        nu_min: Minimum wavenumber (cm⁻¹)
        nu_max: Maximum wavenumber (cm⁻¹)  
        band_width: Width of each spectral band (cm⁻¹)
        spacing: "linear" or "log" - how to distribute band edges
        
    Returns:
        band_edges: Array of [left, right] edges for each band, 
                   shape (nnu_bands, 2)
        
    Raises:
        ValueError: If invalid parameters or spacing mode
        
    Example:
        >>> # Linear spacing (uniform band coverage)
        >>> edges = spectral_band_edges(1000.0, 1200.0, band_width=50.0, spacing="linear")
        >>> # Returns: [[1000, 1050], [1050, 1100], [1100, 1150], [1150, 1200]]
        
        >>> # Log spacing (precise logarithmic distribution)  
        >>> edges = spectral_band_edges(1000.0, 2000.0, band_width=50.0, spacing="log")
        >>> # Returns precisely log-spaced edges
    """
    if nu_min >= nu_max:
        raise ValueError("nu_min must be less than nu_max")
    if band_width <= 0:
        raise ValueError("band_width must be positive")
    if spacing not in ["linear", "log"]:
        raise ValueError("spacing must be 'linear' or 'log'")
    
    if spacing == "linear":
        # Linear spacing: create edges with uniform spacing
        total_range = nu_max - nu_min
        n_bands = int(np.ceil(total_range / band_width))
        
        # Generate edge positions directly
        edge_positions = _set_grid_eslin(nu_min, nu_max, n_bands + 1)
        
        # Create [left, right] pairs
        band_edges = np.column_stack([edge_positions[:-1], edge_positions[1:]])
        
    elif spacing == "log":
        # Logarithmic spacing: create edges with log spacing
        log_range = np.log10(nu_max) - np.log10(nu_min)
        # Estimate number of bands needed
        nu_geometric_mean = np.sqrt(nu_min * nu_max)
        log_spacing_estimate = band_width / (nu_geometric_mean * np.log(10))
        n_bands = max(1, int(np.ceil(log_range / log_spacing_estimate)))
        
        # Generate edge positions directly in log space
        edge_positions = _set_grid_eslog(nu_min, nu_max, n_bands + 1)
        
        # Create [left, right] pairs
        band_edges = np.column_stack([edge_positions[:-1], edge_positions[1:]])
    
    return band_edges


def spectral_bands(nu_min, nu_max, band_width=50.0, spacing="linear"):
    """Generate spectral band centers and edges for banded calculations.
    
    This function computes band centers from precisely calculated edges,
    ensuring mathematical consistency and avoiding precision issues.
    
    Args:
        nu_min: Minimum wavenumber (cm⁻¹)
        nu_max: Maximum wavenumber (cm⁻¹)  
        band_width: Width of each spectral band (cm⁻¹)
        spacing: "linear" or "log" - how to distribute band centers
        
    Returns:
        nu_bands: Band centers, shape (nnu_bands,), ascending order
        band_edges: Band edge positions, shape (nnu_bands, 2)
        
    Example:
        >>> # Linear spacing
        >>> nu_bands, edges = spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="linear")
        >>> # nu_bands: [1025, 1075, 1125, 1175]
        >>> # edges: [[1000, 1050], [1050, 1100], [1100, 1150], [1150, 1200]]
        
        >>> # Log spacing
        >>> nu_bands, edges = spectral_bands(1000.0, 2000.0, band_width=50.0, spacing="log")
        >>> # Returns precisely computed centers and edges from log-spaced calculation
    """
    # Get edges first (primary computation)
    band_edges = spectral_band_edges(nu_min, nu_max, band_width, spacing)
    
    # Compute centers from edges
    if spacing == "linear":
        # Arithmetic mean for linear spacing
        nu_bands = (band_edges[:, 0] + band_edges[:, 1]) / 2.0
    elif spacing == "log":
        # Geometric mean for log spacing
        nu_bands = np.sqrt(band_edges[:, 0] * band_edges[:, 1])
    
    return nu_bands, band_edges


