"""Unit tests for OpaCKD precompute_tables method."""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import config

from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity.opacalc import OpaPremodit
from exojax.opacity.ckd.api import OpaCKD

config.update("jax_enable_x64", True)


class TestPrecomputeTables:
    """Test suite for precompute_tables method."""

    def setup_method(self):
        """Set up test fixtures using mock_mdbExomol."""
        # Setup wavenumber grid and molecular database (small for testing)
        nus, wav, res = mock_wavenumber_grid()
        self.nus = nus
        mdb = mock_mdbExomol("CO")

        # Initialize base opacity calculator
        self.base_opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])

        # Initialize OpaCKD with small parameters for testing
        self.opa_ckd = OpaCKD(
            self.base_opa, Ng=16, band_width=8.0
        )  # Small band width for testing

        # Sample T,P grids
        self.T_grid = jnp.array([990.0, 1010.0])
        self.P_grid = jnp.array([0.032, 0.034])

    def test_precompute_tables_basic(self):
        """Test basic precompute_tables functionality."""
        # Should not raise any errors
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)

        # Check that CKD info was created
        assert self.opa_ckd.ckd_info is not None
        assert self.opa_ckd.ready == True

        # Check basic structure
        assert hasattr(self.opa_ckd.ckd_info, "log_kggrid")
        assert hasattr(self.opa_ckd.ckd_info, "ggrid")
        assert hasattr(self.opa_ckd.ckd_info, "weights")

    def test_precompute_tables_dimensions(self):
        """Test that output dimensions are correct."""
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)

        nT, nP = len(self.T_grid), len(self.P_grid)
        Ng = self.opa_ckd.Ng
        nnu_bands = len(self.opa_ckd.nu_bands)

        # Check log_kggrid dimensions
        expected_shape = (nT, nP, Ng, nnu_bands)
        assert self.opa_ckd.ckd_info.log_kggrid.shape == expected_shape

    def test_validation_errors(self):
        """Test input validation."""
        # Test empty grids
        with pytest.raises(ValueError, match="T_grid and P_grid must not be empty"):
            self.opa_ckd.precompute_tables(jnp.array([]), self.P_grid)

        # Test negative temperatures
        with pytest.raises(ValueError, match="All temperatures must be positive"):
            self.opa_ckd.precompute_tables(jnp.array([-100.0, 1000.0]), self.P_grid)

    def test_average_transmission(self):
        """Test CKD quadrature integration accuracy against direct averaging.
        
        This test validates that the CKD method correctly reproduces the average 
        transmission through each spectral band using Gauss-Legendre quadrature.
        """
        # Pre-compute CKD tables for interpolation
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        Ng = self.opa_ckd.Ng
        nnu_bands = len(self.opa_ckd.nu_bands)

        # Test conditions: intermediate T,P values for interpolation
        Tin = 1000.0  # Temperature between grid points (990, 1010)
        Pin = 0.033   # Pressure between grid points (0.032, 0.034)
        
        # Get CKD cross-section vector and reshape to (Ng, nnu_bands) format
        xsv = self.opa_ckd.xsvector(Tin, Pin)
        xsv_folding = xsv.reshape(Ng, nnu_bands)  # g-ordinates × bands
        
        # Optical depth parameter for transmission calculation
        L = 1.0e22  # Large value to test in optically thick regime
        
        # === CKD METHOD: Gauss-Legendre quadrature integration ===
        # Extract quadrature weights and band information from CKD tables
        weights = self.opa_ckd.ckd_info.weights      # Gauss-Legendre weights
        band_edges = self.opa_ckd.ckd_info.band_edges # Band boundaries [left, right]
        nu_bands = self.opa_ckd.ckd_info.nu_bands     # Band centers
        
        # Compute band-averaged transmission using CKD quadrature:
        # For each band: ∫ exp(-σ*L) dg ≈ Σ(w_i * exp(-σ_i*L))
        ckd_sum = jnp.einsum("n,nm->m", weights, jnp.exp(-xsv_folding * L))

        # === REFERENCE METHOD: Direct fine-grid averaging ===
        # Get high-resolution cross-section vector from base opacity calculator
        xsv_finer = self.base_opa.xsvector(Tin, Pin)
        tau = jnp.exp(-xsv_finer * L)  # Transmission on fine grid
        
        # Compute reference band averages by direct integration over fine grid
        tau_ave = []
        for i in range(len(nu_bands)):
            # Create mask for frequencies within this band's boundaries
            mask = (band_edges[i,0] <= self.nus) * (self.nus < band_edges[i,1])
            # Simple arithmetic average over the band
            tau_ave.append(jnp.mean(tau[mask]))
        tau_ave = jnp.array(tau_ave)
        
        # === ACCURACY VALIDATION ===
        # Compare CKD quadrature results against direct averaging
        # Compute RMS relative error across all bands
        diff = jnp.sqrt(jnp.sum((ckd_sum/tau_ave - 1.0)**2)/len(tau_ave))
        
        # Assert that CKD approximation is accurate to within 0.1%
        assert diff < 0.001
        
        # === OPTIONAL: Visual validation (uncomment to plot) ===
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(211)
        #plt.plot(self.nus, tau, label='Fine grid transmission')
        #plt.plot(nu_bands, tau_ave, 'o', label='Direct band averages')
        #plt.plot(nu_bands, ckd_sum, 's', ls="dashed", label='CKD quadrature')
        #plt.ylabel('Transmission')
        #plt.legend()
        #ax = fig.add_subplot(212)
        #plt.plot(nu_bands, ckd_sum/tau_ave - 1.0, 'ro-')
        #plt.ylabel("Relative error (CKD/direct - 1)")
        #plt.xlabel("Wavenumber (cm⁻¹)")
        #plt.show()


if __name__ == "__main__":
    test_suite = TestPrecomputeTables()
    test_suite.setup_method()

    print("Running precompute_tables tests...")
    test_suite.test_precompute_tables_basic()
    print("✓ Basic functionality test passed")

    test_suite.test_precompute_tables_dimensions()
    print("✓ Dimensions test passed")

    test_suite.test_validation_errors()
    print("✓ Validation test passed")

    test_suite.test_average_transmission()
    print("✓  integration test")

    print("✅ All tests passed!")
