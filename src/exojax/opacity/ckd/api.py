"""API for Correlated-K Distribution (CKD) opacity calculations.

This module provides the OpaCKD class for correlated-k distribution opacity calculations.
CKD enables efficient radiative transfer by reducing the spectral dimensionality while
maintaining accuracy through k-distribution statistical representation.
"""

from __future__ import annotations
from functools import partial
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exojax.opacity.base import OpaCalc


@dataclass(frozen=True)
class CKDTableInfo:
    """Immutable container for CKD table information.
    
    Attributes:
        log_kggrid: Log k-values on g-grid, shape (nT, nP, Ng, nnu_bands)
        ggrid: Gauss-Legendre g-ordinates, shape (Ng,)
        weights: Gauss-Legendre quadrature weights, shape (Ng,)
        T_grid: Temperature grid, shape (nT,)
        P_grid: Pressure grid, shape (nP,)
        nu_bands: Wavenumber band centers, shape (nnu_bands,)
    """
    log_kggrid: jnp.ndarray
    ggrid: jnp.ndarray
    weights: jnp.ndarray
    T_grid: jnp.ndarray
    P_grid: jnp.ndarray
    nu_bands: jnp.ndarray


class OpaCKD(OpaCalc):
    """Opacity Calculator for Correlated-K Distribution (CKD) method.
    
    OpaCKD provides efficient radiative transfer calculations by using pre-computed
    k-distribution tables that statistically represent spectral opacity variations.
    This enables fast atmospheric modeling while maintaining accuracy.
    
    Attributes:
        method: Always "ckd" for this calculator
        base_opa: Underlying opacity calculator (OpaPremodit, OpaModit, etc.)
        Ng: Number of Gauss-Legendre quadrature points
        ckd_info: Pre-computed CKD table information
        ready: Whether calculator is ready for opacity computation
    """
    
    def __init__(
        self,
        base_opa,
        Ng: int = 32,
        nu_bands: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        band_width: float = 50.0,
        band_spacing: str = "log",
        band_overlap_factor: float = 0.0,
    ) -> None:
        """Initialize OpaCKD opacity calculator.
        
        Args:
            base_opa: Base opacity calculator (OpaPremodit, OpaModit, etc.)
            Ng: Number of Gauss-Legendre quadrature points
            nu_bands: Wavenumber bands for CKD table generation.
                     If None, auto-generates from base_opa.nu_grid using band_width
            band_width: Width of each spectral band (cm⁻¹), used if nu_bands is None
            band_spacing: "linear" or "log" spacing for auto-generated bands (default: "log")
            band_overlap_factor: Overlap factor for adjacent bands (0.0-0.5)
        
        Raises:
            ValueError: If base opacity calculator is not ready or invalid parameters
        """
        if not hasattr(base_opa, 'nu_grid'):
            raise ValueError("Base opacity calculator must have nu_grid attribute")
            
        # Initialize parent with base_opa's grid for compatibility
        super().__init__(base_opa.nu_grid)
        
        self.method = "ckd"
        self.base_opa = base_opa
        self.Ng = Ng
        self.band_width = band_width
        self.band_spacing = band_spacing
        self.band_overlap_factor = band_overlap_factor
        
        # Set up spectral bands
        if nu_bands is not None:
            # Use provided bands
            self.nu_bands = jnp.asarray(nu_bands)
        else:
            # Auto-generate bands from base_opa grid
            self._setup_spectral_bands()
        
        # Initialize state
        self.ckd_info = None
        self.ready = False
    
    def _setup_spectral_bands(self) -> None:
        """Set up spectral bands from base opacity grid."""
        from exojax.utils.spectral_bands import spectral_bands
        
        # Get spectral range from base opacity calculator
        nu_min = float(self.base_opa.nu_grid[0])
        nu_max = float(self.base_opa.nu_grid[-1])
        
        # Generate band centers and edges
        nu_bands, band_edges = spectral_bands(
            nu_min=nu_min,
            nu_max=nu_max, 
            band_width=self.band_width,
            spacing=self.band_spacing,
            overlap_factor=self.band_overlap_factor
        )
        
        self.nu_bands = jnp.asarray(nu_bands)
        self.band_edges = jnp.asarray(band_edges)
    
    def _validate_precompute_inputs(
        self, 
        T_grid: jnp.ndarray, 
        P_grid: jnp.ndarray
    ) -> None:
        """Validate inputs for precompute_tables.
        
        Args:
            T_grid: Temperature grid in Kelvin
            P_grid: Pressure grid in bar
            
        Raises:
            ValueError: If validation fails
        """
        # Check base opacity calculator
        if not hasattr(self.base_opa, 'xsmatrix'):
            raise ValueError("Base opacity calculator must have xsmatrix method")
        
        # Validate grid dimensions
        if len(T_grid) == 0 or len(P_grid) == 0:
            raise ValueError("T_grid and P_grid must not be empty")
        
        # Validate physical values
        if jnp.any(T_grid <= 0):
            raise ValueError("All temperatures must be positive")
        
        if jnp.any(P_grid <= 0):
            raise ValueError("All pressures must be positive")
    
    def _process_spectral_band(
        self, 
        i: int, 
        band_edge: jnp.ndarray, 
        xsmatrix_full: jnp.ndarray,
        ggrid: jnp.ndarray, 
        weights: jnp.ndarray,
        compute_ckd_tables
    ) -> Optional[jnp.ndarray]:
        """Process a single spectral band for CKD computation.
        
        Args:
            i: Band index
            band_edge: [left, right] edge positions
            xsmatrix_full: Full cross-section matrix
            ggrid: Gauss-Legendre g-ordinates
            weights: Gauss-Legendre weights
            compute_ckd_tables: CKD computation function
            
        Returns:
            CKD results for this band, or None if band has no coverage
        """
        # Extract wavenumber range for this band using edges
        nu_left, nu_right = band_edge[0], band_edge[1]
        
        # Find indices in base_opa.nu_grid that fall within this band
        mask = (self.base_opa.nu_grid >= nu_left) & (self.base_opa.nu_grid <= nu_right)
        
        if not jnp.any(mask):
            print(f"  Band {i+1}: No coverage, skipping")
            return None
        
        # Extract subgrid cross-sections for this band (no expensive xsmatrix call!)
        # Handle both 2D (nT, nnu) and 3D (nT, nP, nnu) cases
        if len(xsmatrix_full.shape) == 3:
            xsmatrix_band = xsmatrix_full[:, :, mask]
        else:
            xsmatrix_band = xsmatrix_full[:, mask]
        n_freq_band = jnp.sum(mask)
        
        print(f"  Band {i+1}: [{nu_left:.1f}, {nu_right:.1f}] cm⁻¹, {n_freq_band} frequencies")
        
        # Compute CKD for this band
        log_kggrid_band, _, _ = compute_ckd_tables(
            xsmatrix_band, self.Ng
        )
        
        return log_kggrid_band
    
    def precompute_tables(
        self,
        T_grid: Union[np.ndarray, jnp.ndarray],
        P_grid: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Pre-compute CKD tables for given T,P grids.
        
        Args:
            T_grid: Temperature grid in Kelvin
            P_grid: Pressure grid in bar
        """
        # Step 1: Setup and validation
        # Convert to JAX arrays
        T_grid = jnp.asarray(T_grid)
        P_grid = jnp.asarray(P_grid)
        
        # Validate inputs
        self._validate_precompute_inputs(T_grid, P_grid)
        
        # Step 2: Generate Gauss-Legendre g-grid and weights
        from exojax.opacity.ckd.core import gauss_legendre_grid, compute_ckd_tables
        
        ggrid, weights = gauss_legendre_grid(self.Ng)
        
        print(f"Setup complete: T_grid shape {T_grid.shape}, P_grid shape {P_grid.shape}")
        print(f"Generated g-grid: {self.Ng} points, range [{ggrid[0]:.4f}, {ggrid[-1]:.4f}]")
        
        # Step 3: Compute full cross-section matrix once (expensive operation)
        print("Computing full cross-section matrix...")
        xsmatrix_full = self.base_opa.xsmatrix(T_grid, P_grid)
        print(f"Cross-section matrix shape: {xsmatrix_full.shape}")
        
        # Initialize storage for all bands
        nT, nP = len(T_grid), len(P_grid)
        nnu_bands = len(self.nu_bands)
        log_kggrid = jnp.zeros((nT, nP, self.Ng, nnu_bands))
        
        # Process each spectral band using precise edges
        print(f"Processing {nnu_bands} spectral bands...")
        for i, band_edge in enumerate(self.band_edges):
            # Process this band
            log_kggrid_band = self._process_spectral_band(
                i, band_edge, xsmatrix_full, ggrid, weights, compute_ckd_tables
            )
            
            # Store results if band has coverage
            if log_kggrid_band is not None:
                log_kggrid = log_kggrid.at[:, :, :, i].set(log_kggrid_band)
            
        # Step 5: Create CKD table info and finalize
        print("Creating CKD table info...")
        self.ckd_info = CKDTableInfo(
            log_kggrid=log_kggrid,
            ggrid=ggrid,
            weights=weights,
            T_grid=T_grid,
            P_grid=P_grid,
            nu_bands=self.nu_bands
        )
        
        self.ready = True
        print(f"CKD precomputation complete! Ready for interpolation.")
        print(f"Table dimensions: T={len(T_grid)}, P={len(P_grid)}, g={self.Ng}, bands={nnu_bands}")
    
    def xsvector(self, T: float, P: float) -> jnp.ndarray:
        """Compute cross section vector using CKD interpolation.
        
        Args:
            T: Temperature in Kelvin
            P: Pressure in bar
            
        Returns:
            Cross section vector in cm²
            
        Raises:
            ValueError: If CKD tables not pre-computed
        """
        # TODO: Implement xsvector
        pass
    
    def xsmatrix(
        self, 
        T_array: Union[np.ndarray, jnp.ndarray], 
        P_array: Union[np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute cross section matrix using CKD interpolation.
        
        Args:
            T_array: Temperature array in Kelvin
            P_array: Pressure array in bar
            
        Returns:
            Cross section matrix in cm²
        """
        # TODO: Implement xsmatrix
        pass
    
    def save_tables(self, filepath: str) -> None:
        """Save pre-computed CKD tables to file.
        
        Args:
            filepath: Path to save CKD tables
        """
        # TODO: Implement table saving
        pass
    
    def load_tables(self, filepath: str) -> None:
        """Load pre-computed CKD tables from file.
        
        Args:
            filepath: Path to load CKD tables from
        """
        # TODO: Implement table loading
        pass