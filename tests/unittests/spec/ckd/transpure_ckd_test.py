"""unit test for ArtTransPure with OpaCKD"""

from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt import ArtTransPure

config.update("jax_enable_x64", True)


class TestArtTransPureCKD:
    """Unit test for ArtTransPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures"""
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        self.mdb = mock_mdbExomol("H2O")

        self.base_art = ArtTransPure(
            pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=50, integration="simpson"
        )

        self.Tarr = np.linspace(1000.0, 1500.0, 50)
        self.mmr_arr = np.full(50, 0.1)
        self.mean_molecular_weight = np.full(50, 2.33)
        self.radius_btm = 6.9e9
        self.gravity = 2478.57

        self.base_opa = OpaPremodit(self.mdb, nu_grid, auto_trange=[800.0, 1600.0])
        self.opa_ckd = OpaCKD(self.base_opa, Ng=16, band_width=0.5)

    def test_run_ckd(self):
        """test run_ckd method"""
        # Standard calculation
        xsmatrix = self.base_opa.xsmatrix(self.Tarr, self.base_art.pressure)
        dtau = self.base_art.opacity_profile_xs(
            xsmatrix, self.mmr_arr, self.mdb.molmass, self.gravity
        )
        transit_lbl = self.base_art.run(
            dtau, self.Tarr, self.mean_molecular_weight, self.radius_btm, self.gravity
        )

        # Band averages
        transit_avg = []
        band_edges = self.opa_ckd.band_edges
        for band_idx in range(len(self.opa_ckd.nu_bands)):
            mask = (band_edges[band_idx, 0] <= self.nu_grid) & (
                self.nu_grid < band_edges[band_idx, 1]
            )
            transit_avg.append(np.mean(transit_lbl[mask]))
        transit_avg = np.array(transit_avg)

        # CKD calculation
        T_grid = np.linspace(np.min(self.Tarr), np.max(self.Tarr), 10)
        P_grid = np.logspace(
            np.log10(np.min(self.base_art.pressure)),
            np.log10(np.max(self.base_art.pressure)),
            10,
        )

        self.opa_ckd.precompute_tables(T_grid, P_grid)
        xs_ckd = self.opa_ckd.xstensor_ckd(self.Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(
            xs_ckd, self.mmr_arr, self.mdb.molmass, self.gravity
        )
        transit_ckd = self.base_art.run_ckd(
            dtau_ckd,
            self.Tarr,
            self.mean_molecular_weight,
            self.radius_btm,
            self.gravity,
            self.opa_ckd.ckd_info.weights,
        )

        res = np.sqrt(
            np.sum((transit_ckd - transit_avg) ** 2) / len(transit_ckd)
        ) / np.mean(transit_avg)
        assert res < 0.0002  # 0.00011118885289798786 2025/6/24

        """
        # if plotting needed, uncomment below
        import matplotlib.pyplot as plt
        plt.plot(self.nu_grid, transit_lbl, label="Rp2 (LBL by Premodit)", alpha=0.5)
        plt.plot(self.opa_ckd.nu_bands, transit_ckd, label="Rp2 (CKD)")
        plt.plot(self.opa_ckd.nu_bands, transit_avg, label="Rp2 (mean of LBL)")
        plt.legend()
        plt.xlabel("Wavenumber (cm^-1)")
        plt.ylabel("squared radius (normalized)")
        resolution = self.opa_ckd.nu_bands[0]/(band_edges[0, 1] - band_edges[0, 0])
        print("Resolution:", resolution)
        plt.title("CKD Spectrum Comparison (Resolution: {:.2f}, error: {:.5f})".format(resolution, res))
        plt.savefig("ckd_test_transmission"+str(int(resolution))+".png")
        plt.show()
        """


if __name__ == "__main__":
    tt = TestArtTransPureCKD()
    tt.setup_method()
    tt.test_run_ckd()
    print("Transmission CKD test completed successfully.")
