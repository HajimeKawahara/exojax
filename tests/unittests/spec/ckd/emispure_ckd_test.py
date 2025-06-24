"""unit test for ArtEmisPure with OpaPremodit

Note:
    The original file was from integration/unittests_long/premodit/premodit_spectrum_test.py
    These tests takes relatively long time to run. So, one of the database is tested at a time (when pytest).

"""

import pytest
from importlib.resources import files
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure

config.update("jax_enable_x64", True)


from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity.opacalc import OpaPremodit
from exojax.opacity.ckd.api import OpaCKD

config.update("jax_enable_x64", True)


class TestArtEmisPureCKD:
    """ Unit test for ArtEmisPure with OpaCKD"""

    def setup_method(self):
        """Set up test fixtures using mock_mdbExomol."""
        # Setup wavenumber grid and molecular database (small for testing)
        nu_grid, _, _ = mock_wavenumber_grid()
        self.nu_grid = nu_grid
        mdb = mock_mdbExomol("CO")

        self.base_art = ArtEmisPure(
            pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
        )
        self.base_art.change_temperature_range(400.0, 1500.0)
        self.Tarr = self.base_art.powerlaw_temperature(1300.0, 0.1)
        self.mmr_arr = self.base_art.constant_profile(0.1)
        self.gravity = 2478.57
        
        # Initialize base opacity calculator
        self.base_opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1500.0])

        # Initialize OpaCKD with small parameters for testing
        self.opa_ckd = OpaCKD(
            self.base_opa, Ng=16, band_width=8.0
        )  # Small band width for testing



    def test_run_ckd(self):
        """
        """

        # run normal spectrum
        xsmatrix = self.base_opa.xsmatrix(self.Tarr, self.base_art.pressure)
        dtau = self.base_art.opacity_profile_xs(xsmatrix, self.mmr_arr, self.base_opa.mdb.molmass, self.gravity)
        F0 = self.base_art.run(dtau, self.Tarr)

        # run CKD spectrum
        NTgrid = 10        
        self.T_grid = np.linspace(np.min(self.Tarr), np.max(self.Tarr), NTgrid)
        NPgrid = 10
        self.P_grid = np.logspace(np.log10(np.min(self.base_art.pressure)), np.log10(np.max(self.base_art.pressure)), NPgrid)
        self.opa_ckd.precompute_tables(self.T_grid, self.P_grid)
        
        xs_ckd = self.opa_ckd.xstensor_ckd(self.Tarr, self.base_art.pressure)
        dtau_ckd = self.base_art.opacity_profile_xs_ckd(xs_ckd, self.mmr_arr, self.base_opa.mdb.molmass, self.gravity) #shape is (100,16,5)
        F0_ckd = self.base_art.run_ckd(dtau_ckd, self.Tarr, self.opa_ckd.ckd_info.weights, self.opa_ckd.nu_bands)

if __name__ == "__main__":
    tt = TestArtEmisPureCKD()
    tt.setup_method()
    tt.test_run_ckd()
    print("CKD test completed successfully.")