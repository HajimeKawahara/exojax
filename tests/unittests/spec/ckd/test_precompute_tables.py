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
        nus, wav, res = mock_wavenumber_grid(lambda0=22930.0, lambda1=22940.0, Nx=1000)
        mdb = mock_mdbExomol("CO")
        
        # Initialize base opacity calculator
        self.base_opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])
        
        # Initialize OpaCKD with small parameters for testing
        self.opa_ckd = OpaCKD(self.base_opa, Ng=8, band_width=2.0)  # Small band width for testing
        
        # Sample T,P grids
        self.T_grid = jnp.array([800.0, 1200.0])
        self.P_grid = jnp.array([0.01, 0.1])
    
    def test_precompute_tables_basic(self):
        """Test basic precompute_tables functionality."""
        # Should not raise any errors
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        
        # Check that CKD info was created
        assert self.opa_ckd.ckd_info is not None
        assert self.opa_ckd.ready == True
        
        # Check basic structure
        assert hasattr(self.opa_ckd.ckd_info, 'log_kggrid')
        assert hasattr(self.opa_ckd.ckd_info, 'ggrid')
        assert hasattr(self.opa_ckd.ckd_info, 'weights')
    
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
    
    print("✅ All tests passed!")