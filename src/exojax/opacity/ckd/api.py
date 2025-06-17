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
        
        # Generate band centers
        nu_bands = spectral_bands(
            nu_min=nu_min,
            nu_max=nu_max, 
            band_width=self.band_width,
            spacing=self.band_spacing,
            overlap_factor=self.band_overlap_factor
        )
        
        self.nu_bands = jnp.asarray(nu_bands)
    
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
        # TODO: Implement table pre-computation
        pass
    
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