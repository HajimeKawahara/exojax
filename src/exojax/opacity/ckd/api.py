"""API for Correlated-K Distribution (CKD) opacity calculations.

This module provides the OpaCKD class for correlated-k distribution opacity calculations.
CKD enables efficient radiative transfer by reducing the spectral dimensionality while
maintaining accuracy through k-distribution statistical representation.
"""

from __future__ import annotations
from typing import Union, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import vmap

from exojax.opacity.base import OpaCalc
from exojax.opacity.ckd.core import gauss_legendre_grid
from exojax.opacity.ckd.core import compute_ckd_tables
from exojax.opacity.ckd.core import interpolate_log_k_2d
from exojax.utils.spectral_bands import spectral_bands


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
        band_edges: Wavenumber band edges, shape (nnu_bands, 2)
    """

    log_kggrid: jnp.ndarray
    ggrid: jnp.ndarray
    weights: jnp.ndarray
    T_grid: jnp.ndarray
    P_grid: jnp.ndarray
    nu_bands: jnp.ndarray
    band_edges: jnp.ndarray


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
        band_width: float = 50.0,
        band_spacing: str = "log",
    ) -> None:
        """Initialize OpaCKD opacity calculator.

        Args:
            base_opa: Base opacity calculator (OpaPremodit, OpaModit, etc.)
            Ng: Number of Gauss-Legendre quadrature points
            band_width: Width of each spectral band (cm⁻¹)
            band_spacing: "linear" or "log" spacing for band generation (default: "log")

        Raises:
            ValueError: If base opacity calculator is not ready or invalid parameters
        """
        if not hasattr(base_opa, "nu_grid"):
            raise ValueError("Base opacity calculator must have nu_grid attribute")

        # Initialize parent with base_opa's grid for compatibility
        super().__init__(base_opa.nu_grid)

        self.method = "ckd"
        self.base_opa = base_opa
        self.Ng = Ng
        self.band_width = band_width
        self.band_spacing = band_spacing

        # Auto-generate spectral bands from base_opa grid
        self._setup_spectral_bands()

        # Initialize state
        self.ckd_info = None
        self.ready = False

    def _setup_spectral_bands(self) -> None:
        """Set up spectral bands from base opacity grid."""

        # Get spectral range from base opacity calculator
        nu_min = float(self.base_opa.nu_grid[0])
        nu_max = float(self.base_opa.nu_grid[-1])

        # Generate band centers and edges
        nu_bands, band_edges = spectral_bands(
            nu_min=nu_min,
            nu_max=nu_max,
            band_width=self.band_width,
            spacing=self.band_spacing,
        )

        self.nu_bands = jnp.asarray(nu_bands)
        self.band_edges = jnp.asarray(band_edges)

    def _validate_precompute_inputs(
        self, T_grid: jnp.ndarray, P_grid: jnp.ndarray
    ) -> None:
        """Validate inputs for precompute_tables.

        Args:
            T_grid: Temperature grid in Kelvin
            P_grid: Pressure grid in bar

        Raises:
            ValueError: If validation fails
        """
        # Check base opacity calculator
        if not hasattr(self.base_opa, "xsmatrix"):
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
        compute_ckd_tables,
    ) -> Optional[jnp.ndarray]:
        """Process a single spectral band for CKD computation.

        Args:
            i: Band index
            band_edge: [left, right] edge positions
            xsmatrix_full: Full cross-section matrix
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

        print(
            f"  Band {i+1}: [{nu_left:.1f}, {nu_right:.1f}] cm⁻¹, {n_freq_band} frequencies"
        )

        # Compute CKD for this band
        log_kggrid_band, _, _ = compute_ckd_tables(xsmatrix_band, self.Ng)

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

        self._validate_precompute_inputs(T_grid, P_grid)
        ggrid, weights = gauss_legendre_grid(self.Ng)
        print(
            f"Generated g-grid: {self.Ng} points, range [{ggrid[0]:.4f}, {ggrid[-1]:.4f}]"
        )
        xsmatrix_full = self.base_opa.xsmatrix(T_grid, P_grid)

        # Initialize storage for all bands
        nT, nP = len(T_grid), len(P_grid)
        nnu_bands = len(self.nu_bands)
        log_kggrid = jnp.zeros((nT, nP, self.Ng, nnu_bands))

        # Process each spectral band using precise edges
        print(f"Processing {nnu_bands} spectral bands...")
        for i, band_edge in enumerate(self.band_edges):
            # Process this band
            log_kggrid_band = self._process_spectral_band(
                i, band_edge, xsmatrix_full, compute_ckd_tables
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
            nu_bands=self.nu_bands,
            band_edges=self.band_edges,
        )

        self.ready = True
        print(f"CKD precomputation complete! Ready for interpolation.")
        print(
            f"Table dimensions: T={len(T_grid)}, P={len(P_grid)}, g={self.Ng}, bands={nnu_bands}"
        )

    def _interpolate_log_k(self, T: float, P: float) -> jnp.ndarray:
        """JAX-compatible 2D interpolation of log_kggrid at given T,P.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar

        Returns:
            Interpolated log k-values, shape (Ng, nnu_bands)
        """
        return interpolate_log_k_2d(
            self.ckd_info.log_kggrid, self.ckd_info.T_grid, self.ckd_info.P_grid, T, P
        )

    def xsarray_ckd(self, T: float, P: float) -> jnp.ndarray:
        """Compute CKD cross section array using interpolation.

        Interpolates pre-computed CKD tables at given T,P and returns the 2D array
        with shape (Ng, nnu_bands) containing g-ordinates and spectral bands.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar

        Returns:
            Cross section array in cm², shape (Ng, nnu_bands)
            First dimension: g-ordinates (quadrature points)
            Second dimension: spectral bands

        """
        log_k_interp = self._interpolate_log_k(T, P)  # Shape: (Ng, nnu_bands)
        return jnp.exp(log_k_interp)

    def xstensor_ckd(
        self,
        T_array: Union[np.ndarray, jnp.ndarray],
        P_array: Union[np.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute CKD cross section tensor using interpolation.

        Computes CKD cross-sections for paired (T,P) values: (T1,P1), (T2,P2), ...
        Returns a 3D tensor with layers, g-ordinates, and spectral bands.

        Args:
            T_array: Temperature array in Kelvin, shape (Nlayer,)
            P_array: Pressure array in bar, shape (Nlayer,)

        Returns:
            Cross section tensor in cm², shape (Nlayer, Ng, nnu_bands)
            First dimension: atmospheric layers
            Second dimension: g-ordinates (quadrature points)
            Third dimension: spectral bands

        """
        xsarray_vmap = vmap(self.xsarray_ckd, in_axes=(0, 0))
        return xsarray_vmap(T_array, P_array)
