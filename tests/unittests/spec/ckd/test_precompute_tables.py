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

    def test_xsmatrix_method(self):
        """Test xsmatrix method with paired (T,P) values."""
        # Pre-compute CKD tables
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        
        # Test with paired (T,P) values
        T_test = jnp.array([995.0, 1005.0, 1000.0])  # 3 temperature values
        P_test = jnp.array([0.033, 0.034, 0.0325])   # 3 corresponding pressure values
        
        # Call xsmatrix
        xsmatrix_result = self.opa_ckd.xsmatrix(T_test, P_test)
        
        # Check output shape
        Nlayer = len(T_test)
        expected_cols = self.opa_ckd.Ng * len(self.opa_ckd.nu_bands)
        expected_shape = (Nlayer, expected_cols)
        assert xsmatrix_result.shape == expected_shape
        
        # Verify consistency with xsvector for each (T,P) pair
        for i in range(len(T_test)):
            xsv_individual = self.opa_ckd.xsvector(T_test[i], P_test[i])
            xsv_from_matrix = xsmatrix_result[i, :]
            assert jnp.allclose(xsv_individual, xsv_from_matrix, rtol=1e-10)

    def test_xsmatrix_average_transmission(self):
        """Test xsmatrix batch transmission calculation accuracy against reference.
        
        This test validates that xsmatrix produces accurate transmission results
        when used for multiple atmospheric layers, comparing against both individual
        xsvector calls and direct fine-grid averaging for each layer.
        """
        # Pre-compute CKD tables for interpolation
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        Ng = self.opa_ckd.Ng
        nnu_bands = len(self.opa_ckd.nu_bands)

        # Define multiple atmospheric layers with different (T,P) conditions
        T_layers = jnp.array([995.0, 1000.0, 1005.0])  # 3 atmospheric layers
        P_layers = jnp.array([0.033, 0.034, 0.032])    # Corresponding pressures
        
        # Optical depth parameter for transmission calculation
        L = 1.0e22  # Large value to test in optically thick regime
        
        # === METHOD 1: Batch xsmatrix calculation ===
        # Get cross-section matrix for all layers at once
        xsmatrix_result = self.opa_ckd.xsmatrix(T_layers, P_layers)  # Shape: (3, Ng*nnu_bands)
        
        # Reshape to (Nlayers, Ng, nnu_bands) for transmission calculation
        xsmatrix_folded = xsmatrix_result.reshape(len(T_layers), Ng, nnu_bands)
        
        # Extract CKD quadrature information
        weights = self.opa_ckd.ckd_info.weights      # Gauss-Legendre weights
        band_edges = self.opa_ckd.ckd_info.band_edges # Band boundaries
        
        # Compute batch transmission using CKD quadrature for each layer
        # For each layer and band: ∫ exp(-σ*L) dg ≈ Σ(w_i * exp(-σ_i*L))
        ckd_batch = jnp.einsum("n,lnm->lm", weights, jnp.exp(-xsmatrix_folded * L))
        
        # === METHOD 2: Individual xsvector calculations (reference) ===
        ckd_individual = []
        for i, (T, P) in enumerate(zip(T_layers, P_layers)):
            # Get individual cross-section vector and reshape
            xsv = self.opa_ckd.xsvector(T, P)
            xsv_folded = xsv.reshape(Ng, nnu_bands)
            
            # Compute transmission for this layer
            ckd_layer = jnp.einsum("n,nm->m", weights, jnp.exp(-xsv_folded * L))
            ckd_individual.append(ckd_layer)
        
        ckd_individual = jnp.array(ckd_individual)  # Shape: (Nlayers, nnu_bands)
        
        # === VALIDATION 1: Batch vs Individual consistency ===
        # Verify that batch xsmatrix gives identical results to individual xsvector calls
        assert jnp.allclose(ckd_batch, ckd_individual, rtol=1e-12)
        
        # === VALIDATION 2: CKD accuracy against fine-grid reference ===
        # Compare against direct fine-grid averaging for each layer
        max_error_across_layers = 0.0
        
        for layer_idx, (T, P) in enumerate(zip(T_layers, P_layers)):
            # Get fine-grid cross-section for this layer
            xsv_fine = self.base_opa.xsvector(T, P)
            tau_fine = jnp.exp(-xsv_fine * L)  # Fine-grid transmission
            
            # Compute reference band averages by direct integration
            tau_reference = []
            for band_idx in range(nnu_bands):
                # Create mask for frequencies within this band
                mask = (band_edges[band_idx,0] <= self.nus) * (self.nus < band_edges[band_idx,1])
                # Arithmetic average over the band
                tau_reference.append(jnp.mean(tau_fine[mask]))
            tau_reference = jnp.array(tau_reference)
            
            # Compare CKD result for this layer against reference
            ckd_layer_result = ckd_batch[layer_idx, :]
            layer_error = jnp.sqrt(jnp.sum((ckd_layer_result/tau_reference - 1.0)**2)/nnu_bands)
            max_error_across_layers = max(max_error_across_layers, layer_error)
        
        # Assert that CKD approximation is accurate across all layers
        assert max_error_across_layers < 0.005  # 0.5% accuracy for all layers (relaxed for multi-layer test)
        
        # === OPTIONAL: Verify shapes and basic properties ===
        assert ckd_batch.shape == (len(T_layers), nnu_bands)
        assert jnp.all(ckd_batch > 0)  # Transmission should be positive
        assert jnp.all(ckd_batch <= 1)  # Transmission should be ≤ 1


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
    print("✓ Integration test passed")

    test_suite.test_xsmatrix_method()
    print("✓ xsmatrix test passed")

    test_suite.test_xsmatrix_average_transmission()
    print("✓ xsmatrix transmission test passed")

    print("✅ All tests passed!")
