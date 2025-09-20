"""unit test for ArtReflectPure with OpaCKD

Test CKD implementation for pure reflection RT
"""

from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt import ArtReflectPure

config.update("jax_enable_x64", True)


class TestArtReflectPureCKD:
    """Unit test for ArtReflectPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures using mock_mdbExomol."""
        # Setup wavenumber grid and molecular database (small for testing)
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        self.mdb = mock_mdbExomol("H2O")

        self.base_art = ArtReflectPure(
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

        # Set up scattering parameters (constant for simplicity)
        self.single_scattering_albedo = np.full((100, len(nu_grid)), 0.8)
        self.asymmetric_parameter = np.full((100, len(nu_grid)), 0.6)

        # Set up surface and incoming flux parameters
        self.reflectivity_surface = np.full(len(nu_grid), 0.3)
        self.incoming_flux = np.ones(len(nu_grid))  # Normalized incoming flux

    def test_run_ckd(self):
        """test run_ckd method of ArtReflectPure with OpaCKD."""

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

        # Prepare scattering parameters for CKD (average over bands)
        band_edges = self.opa_ckd.band_edges
        ssa_ckd = np.zeros((100, len(self.opa_ckd.nu_bands)))
        g_ckd = np.zeros((100, len(self.opa_ckd.nu_bands)))

        for band_idx in range(len(self.opa_ckd.nu_bands)):
            # Create mask for frequencies within this band
            mask = (band_edges[band_idx, 0] <= self.nu_grid) & (
                self.nu_grid < band_edges[band_idx, 1]
            )
            # Average scattering parameters over the band
            ssa_ckd[:, band_idx] = np.mean(
                self.single_scattering_albedo[:, mask], axis=1
            )
            g_ckd[:, band_idx] = np.mean(self.asymmetric_parameter[:, mask], axis=1)

        # Prepare surface parameters for CKD (average over bands)
        reflectivity_ckd = np.zeros(len(self.opa_ckd.nu_bands))
        incoming_flux_ckd = np.zeros(len(self.opa_ckd.nu_bands))

        for band_idx in range(len(self.opa_ckd.nu_bands)):
            mask = (band_edges[band_idx, 0] <= self.nu_grid) * (
                self.nu_grid < band_edges[band_idx, 1]
            )
            reflectivity_ckd[band_idx] = np.mean(self.reflectivity_surface[mask])
            incoming_flux_ckd[band_idx] = np.mean(self.incoming_flux[mask])

        # Run CKD reflection
        R_ckd = self.base_art.run_ckd(
            dtau_ckd,
            ssa_ckd,
            g_ckd,
            reflectivity_ckd,
            incoming_flux_ckd,
            self.opa_ckd.ckd_info.weights,
        )

        # Basic validation - check output shape and no NaN values
        assert R_ckd.shape == (len(self.opa_ckd.nu_bands),)
        assert not np.any(np.isnan(R_ckd))
        assert np.all(R_ckd >= 0)  # Reflected flux should be non-negative

        print(f"CKD reflection test passed! Output shape: {R_ckd.shape}")
        print(f"Reflection range: [{np.min(R_ckd):.2e}, {np.max(R_ckd):.2e}]")


if __name__ == "__main__":
    tt = TestArtReflectPureCKD()
    tt.setup_method()
    tt.test_run_ckd()
    print("ArtReflectPure CKD test completed successfully.")
