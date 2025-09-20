"""unit test for ArtAbsPure with OpaCKD

Test CKD implementation for pure absorption RT
"""

from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt.reflect import ArtAbsPure

config.update("jax_enable_x64", True)


class TestArtAbsPureCKD:
    """Unit test for ArtAbsPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures using mock_mdbExomol."""
        # Setup wavenumber grid and molecular database (small for testing)
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        self.mdb = mock_mdbExomol("H2O")

        self.base_art = ArtAbsPure(
            pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
        )
        self.mmr_arr = np.full(100, 0.1)  # Constant mixing ratio
        self.gravity = 2478.57

        # Initialize base opacity calculator
        self.base_opa = OpaPremodit(self.mdb, nu_grid, auto_trange=[500.0, 1500.0])

        # Initialize OpaCKD with small parameters for testing
        self.opa_ckd = OpaCKD(
            self.base_opa, Ng=16, band_width=1.0
        )  # Small band width for testing

        # Set up absorption parameters
        self.pressure_surface = 1.0  # bar
        self.incoming_flux = np.ones(len(nu_grid))  # Normalized incoming flux
        self.mu_in = 0.5  # Cosine of incoming angle
        self.mu_out = 0.8  # Cosine of outgoing angle (None for ground observer)

    def test_run_ckd(self):
        """test run_ckd method of ArtAbsPure with OpaCKD."""

        # Pre-compute CKD tables
        NTgrid = 8
        Tarr = np.linspace(1000.0, 1500.0, 100)  # Temperature for opacity calculation
        T_grid = np.linspace(np.min(Tarr), np.max(Tarr), NTgrid)
        NPgrid = 8
        P_grid = np.logspace(
            np.log10(np.min(self.base_art.pressure)),
            np.log10(np.max(self.base_art.pressure)),
            NPgrid,
        )
        self.opa_ckd.precompute_tables(T_grid, P_grid)

        # Get CKD optical depth tensor
        xs_ckd = self.opa_ckd.xstensor_ckd(Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(
            xs_ckd, self.mmr_arr, self.mdb.molmass, self.gravity
        )

        # Prepare incoming flux for CKD (average over bands)
        band_edges = self.opa_ckd.band_edges
        incoming_flux_ckd = np.zeros(len(self.opa_ckd.nu_bands))

        for band_idx in range(len(self.opa_ckd.nu_bands)):
            mask = (band_edges[band_idx, 0] <= self.nu_grid) & (
                self.nu_grid < band_edges[band_idx, 1]
            )
            incoming_flux_ckd[band_idx] = np.mean(self.incoming_flux[mask])

        # Run CKD absorption
        A_ckd = self.base_art.run_ckd(
            dtau_ckd,
            self.pressure_surface,
            incoming_flux_ckd,
            self.mu_in,
            self.mu_out,
            self.opa_ckd.ckd_info.weights,
        )

        # Basic validation - check output shape and no NaN values
        assert A_ckd.shape == (len(self.opa_ckd.nu_bands),)
        assert not np.any(np.isnan(A_ckd))
        assert np.all(A_ckd >= 0)  # Transmitted flux should be non-negative
        assert np.all(
            A_ckd <= np.max(incoming_flux_ckd)
        )  # Should not exceed incoming flux

        print(f"CKD absorption test passed! Output shape: {A_ckd.shape}")
        print(f"Transmission range: [{np.min(A_ckd):.2e}, {np.max(A_ckd):.2e}]")

    def test_run_ckd_ground_observer(self):
        """test run_ckd method with ground observer (mu_out=None)."""

        # Pre-compute CKD tables
        NTgrid = 8
        Tarr = np.linspace(1000.0, 1500.0, 100)
        T_grid = np.linspace(np.min(Tarr), np.max(Tarr), NTgrid)
        NPgrid = 8
        P_grid = np.logspace(
            np.log10(np.min(self.base_art.pressure)),
            np.log10(np.max(self.base_art.pressure)),
            NPgrid,
        )
        self.opa_ckd.precompute_tables(T_grid, P_grid)

        # Get CKD optical depth tensor
        xs_ckd = self.opa_ckd.xstensor_ckd(Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(
            xs_ckd, self.mmr_arr, self.mdb.molmass, self.gravity
        )

        # Prepare incoming flux for CKD
        band_edges = self.opa_ckd.band_edges
        incoming_flux_ckd = np.zeros(len(self.opa_ckd.nu_bands))

        for band_idx in range(len(self.opa_ckd.nu_bands)):
            mask = (band_edges[band_idx, 0] <= self.nu_grid) * (
                self.nu_grid < band_edges[band_idx, 1]
            )
            incoming_flux_ckd[band_idx] = np.mean(self.incoming_flux[mask])

        # Run CKD absorption with ground observer
        A_ckd_ground = self.base_art.run_ckd(
            dtau_ckd,
            self.pressure_surface,
            incoming_flux_ckd,
            self.mu_in,
            None,
            self.opa_ckd.ckd_info.weights,
        )

        # Basic validation
        assert A_ckd_ground.shape == (len(self.opa_ckd.nu_bands),)
        assert not np.any(np.isnan(A_ckd_ground))
        assert np.all(A_ckd_ground >= 0)

        print(f"CKD absorption (ground observer) test passed!")


if __name__ == "__main__":
    tt = TestArtAbsPureCKD()
    tt.setup_method()
    tt.test_run_ckd()
    tt.test_run_ckd_ground_observer()
    print("ArtAbsPure CKD test completed successfully.")
