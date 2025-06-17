"""Spectral band and subgrid utilities.

This module provides utilities for generating spectral bands and extracting
subgrids from full wavenumber grids. These functions are useful for:
- Correlated-K Distribution (CKD) computations
- Memory-efficient opacity calculations
- Spectral chunking and segmentation
- Instrumental band analysis
"""

import numpy as np


def spectral_bands(nu_min, nu_max, band_width=50.0, spacing="linear", overlap_factor=0.0):
    """Generate spectral band centers for banded calculations.
    
    Creates spectral bands covering the range [nu_min, nu_max] with either
    linear or logarithmic spacing of band centers.
    
    Args:
        nu_min: Minimum wavenumber (cm⁻¹)
        nu_max: Maximum wavenumber (cm⁻¹)  
        band_width: Width of each spectral band (cm⁻¹)
        spacing: "linear" or "log" - how to distribute band centers
        overlap_factor: Fraction of band_width to overlap adjacent bands (0.0-0.5)
        
    Returns:
        nu_bands: Band centers, shape (nnu_bands,), ascending order
        
    Raises:
        ValueError: If invalid parameters or spacing mode
        
    Example:
        >>> # Linear spacing (uniform band coverage)
        >>> nu_bands = spectral_bands(1000.0, 1200.0, band_width=50.0, spacing="linear")
        >>> # Returns: [1025, 1075, 1125, 1175]
        
        >>> # Log spacing (more bands at lower wavenumbers)  
        >>> nu_bands = spectral_bands(1000.0, 2000.0, band_width=50.0, spacing="log")
        >>> # Returns denser spacing near 1000 cm⁻¹, sparser near 2000 cm⁻¹
    """
    if nu_min >= nu_max:
        raise ValueError("nu_min must be less than nu_max")
    if band_width <= 0:
        raise ValueError("band_width must be positive")
    if overlap_factor < 0 or overlap_factor >= 0.5:
        raise ValueError("overlap_factor must be in range [0, 0.5)")
    if spacing not in ["linear", "log"]:
        raise ValueError("spacing must be 'linear' or 'log'")
    
    # Calculate effective spacing between band centers
    band_spacing = band_width * (1.0 - overlap_factor)
    
    if spacing == "linear":
        # Linear spacing: uniform distribution of band centers
        total_range = nu_max - nu_min
        n_bands = int(np.ceil(total_range / band_spacing)) + 1
        
        # Generate linearly spaced band centers
        band_centers = nu_min + band_width/2 + np.arange(n_bands) * band_spacing
        
        # Keep only bands whose centers are within the extended range
        valid_mask = band_centers <= (nu_max + band_width/2)
        band_centers = band_centers[valid_mask]
        
    elif spacing == "log":
        # Logarithmic spacing: more bands at lower wavenumbers
        # Calculate number of bands based on logarithmic scale
        log_range = np.log10(nu_max) - np.log10(nu_min)
        # Estimate spacing in log space to achieve roughly band_spacing in linear space
        # Use geometric mean as representative wavenumber for spacing estimate
        nu_geometric_mean = np.sqrt(nu_min * nu_max)
        log_spacing_estimate = band_spacing / (nu_geometric_mean * np.log(10))
        n_bands = int(np.ceil(log_range / log_spacing_estimate)) + 1
        
        # Generate logarithmically spaced band centers
        # Start from nu_min + band_width/2 to ensure first band covers nu_min
        log_start = np.log10(nu_min + band_width/2)
        log_end = np.log10(nu_max + band_width/2)  # Extended to ensure coverage
        band_centers = np.logspace(log_start, log_end, n_bands)
        
        # Filter to keep only bands that provide meaningful coverage
        # Remove bands whose centers are too close to boundaries
        valid_mask = (band_centers >= nu_min + band_width/4) & (band_centers <= nu_max + band_width/4)
        band_centers = band_centers[valid_mask]
    
    return band_centers