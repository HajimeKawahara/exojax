"""unit test for ArtTransPure with OpaCKD"""

from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt import ArtTransPure
from exojax.test.data import get_testdata_filename, TESTDATA_CO_EXOMOL_PREMODIT_TRANSMISSION_REF

config.update("jax_enable_x64", True)


class TestArtTransPureCKD:
    """Unit test for ArtTransPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures"""
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        mdb = mock_mdbExomol("CO")

        self.base_art = ArtTransPure(
            pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=50, integration="simpson"
        )
        
        self.Tarr = np.linspace(1000.0, 1500.0, 50)
        self.mmr_arr = np.full(50, 0.1)
        self.mean_molecular_weight = np.full(50, 2.33)
        self.radius_btm = 6.9e9
        self.gravity = 2478.57

        self.base_opa = OpaPremodit(mdb, nu_grid, auto_trange=[800.0, 1600.0])
        self.opa_ckd = OpaCKD(self.base_opa, Ng=16, band_width=0.5)

    def test_run_ckd_vs_reference(self):
        """test run_ckd against reference data"""
        # Load reference data
        fn = get_testdata_filename(TESTDATA_CO_EXOMOL_PREMODIT_TRANSMISSION_REF)
        dat = np.loadtxt(fn, delimiter=",")
        transit_ref = dat[:, 1]

        # Run standard calculation
        xsmatrix = self.base_opa.xsmatrix(self.Tarr, self.base_art.pressure)
        dtau = self.base_art.opacity_profile_xs(xsmatrix, self.mmr_arr, self.base_opa.mdb.molmass, self.gravity)
        F0 = self.base_art.run(dtau, self.Tarr, self.mean_molecular_weight, self.radius_btm, self.gravity)
        
        # Compare with reference
        relative_diff = np.abs((F0 - transit_ref) / transit_ref)
        max_relative_diff = np.max(relative_diff)
        
        assert max_relative_diff < 1e-4

    def test_run_ckd(self):
        """test run_ckd method"""
        # Standard calculation
        xsmatrix = self.base_opa.xsmatrix(self.Tarr, self.base_art.pressure)
        dtau = self.base_art.opacity_profile_xs(xsmatrix, self.mmr_arr, self.base_opa.mdb.molmass, self.gravity)
        F0 = self.base_art.run(dtau, self.Tarr, self.mean_molecular_weight, self.radius_btm, self.gravity)
        
        # Band averages
        transit_avg = []
        band_edges = self.opa_ckd.band_edges
        for band_idx in range(len(self.opa_ckd.nu_bands)):
            mask = (band_edges[band_idx, 0] <= self.nu_grid) * (self.nu_grid < band_edges[band_idx, 1])
            transit_avg.append(np.mean(F0[mask]))
        transit_avg = np.array(transit_avg)

        # CKD calculation
        T_grid = np.linspace(np.min(self.Tarr), np.max(self.Tarr), 10)
        P_grid = np.logspace(np.log10(np.min(self.base_art.pressure)), np.log10(np.max(self.base_art.pressure)), 10)
        
        self.opa_ckd.precompute_tables(T_grid, P_grid)
        xs_ckd = self.opa_ckd.xstensor_ckd(self.Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(xs_ckd, self.mmr_arr, self.base_opa.mdb.molmass, self.gravity)
        F0_ckd = self.base_art.run_ckd(dtau_ckd, self.Tarr, self.mean_molecular_weight, self.radius_btm, self.gravity,
                                       self.opa_ckd.ckd_info.weights)

        res = np.sqrt(np.sum((F0_ckd - transit_avg)**2)/len(F0_ckd))/np.mean(transit_avg)
        assert res < 0.05


if __name__ == "__main__":
    tt = TestArtTransPureCKD()
    tt.setup_method()
    tt.test_run_ckd_vs_reference()
    tt.test_run_ckd()
    print("Transmission CKD test completed successfully.")