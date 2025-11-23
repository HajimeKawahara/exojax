"""unit test for ArtEmisPure with OpaPremodit

Note:
    The original file was from integration/unittests_long/premodit/premodit_spectrum_test.py
    These tests takes relatively long time to run. So, one of the database is tested at a time (when pytest).

"""

from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD
from exojax.opacity import OpaPremodit
from exojax.rt import ArtEmisPure

config.update("jax_enable_x64", True)


class TestArtEmisPureCKD:
    """Unit test for ArtEmisPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures using mock_mdbExomol."""
        # Setup wavenumber grid and molecular database (small for testing)
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        self.mdb = mock_mdbExomol("H2O")

        self.base_art = ArtEmisPure(
            pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
        )
        self.base_art.change_temperature_range(400.0, 1500.0)
        self.Tarr = self.base_art.powerlaw_temperature(1300.0, 0.1)
        self.mmr_arr = self.base_art.constant_profile(0.1)
        self.gravity = 2478.57

        # Initialize base opacity calculator
        self.base_opa = OpaPremodit(self.mdb, nu_grid, auto_trange=[500.0, 1500.0])

        # Initialize OpaCKD with small parameters for testing
        self.opa_ckd = OpaCKD(
            self.base_opa, Ng=32, band_width=0.5
        )  # Small band width for testing

    def test_run_ckd(self):
        """test run_ckd method of ArtEmisPure with OpaCKD."""

        # run normal spectrum
        xsmatrix = self.base_opa.xsmatrix(self.Tarr, self.base_art.pressure)
        dtau = self.base_art.opacity_profile_xs(
            xsmatrix, self.mmr_arr, self.mdb.molmass, self.gravity
        )
        F0 = self.base_art.run(dtau, self.Tarr)
        # Compute reference band averages by direct integration
        flux_average_reference = []
        band_edges = self.opa_ckd.band_edges
        for band_idx in range(len(self.opa_ckd.nu_bands)):
            # Create mask for frequencies within this band
            mask = (band_edges[band_idx, 0] <= self.nu_grid) & (
                self.nu_grid < band_edges[band_idx, 1]
            )
            # Arithmetic average over the band
            flux_average_reference.append(np.mean(F0[mask]))
        flux_average_reference = np.array(flux_average_reference)

        # run CKD spectrum
        NTgrid = 10
        self.T_grid = np.linspace(np.min(self.Tarr), np.max(self.Tarr), NTgrid)
        NPgrid = 10
        self.P_grid = np.logspace(
            np.log10(np.min(self.base_art.pressure)),
            np.log10(np.max(self.base_art.pressure)),
            NPgrid,
        )
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        xs_ckd = self.opa_ckd.xstensor_ckd(self.Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(
            xs_ckd, self.mmr_arr, self.mdb.molmass, self.gravity
        )  # shape is (100,16,5)
        F0_ckd = self.base_art.run_ckd(
            dtau_ckd, self.Tarr, self.opa_ckd.ckd_info.weights, self.opa_ckd.nu_bands
        )

        res = np.sqrt(
            np.sum((F0_ckd - flux_average_reference) ** 2) / len(F0_ckd)
        ) / np.mean(flux_average_reference)

        assert res < 0.05  # 0.04067868171246608 2025/6/24

        """
        # if plotting needed, uncomment below
        import matplotlib.pyplot as plt
        plt.plot(self.nu_grid, F0, label="F0 (LBL by Premodit)", alpha=0.5)
        plt.plot(self.opa_ckd.nu_bands, F0_ckd, label="F0 (CKD)")
        plt.plot(self.opa_ckd.nu_bands, flux_average_reference, label="F0 (mean of LBL)")
        plt.legend()
        plt.xlabel("Wavenumber (cm^-1)")
        plt.ylabel("absolute Flux")
        resolution = self.opa_ckd.nu_bands[0]/(band_edges[0, 1] - band_edges[0, 0])
        print("Resolution:", resolution)
        plt.title("CKD Spectrum Comparison (Resolution: {:.2f}, error: {:.4f})".format(resolution, res))
        plt.savefig("ckd_test_spectrum"+str(int(resolution))+".png")
        plt.show()
        """


if __name__ == "__main__":
    tt = TestArtEmisPureCKD()
    tt.setup_method()
    tt.test_run_ckd()
    print("CKD test completed successfully.")
