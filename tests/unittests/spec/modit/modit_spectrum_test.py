"""short integration tests for MODIT spectrum"""

import pytest
from importlib.resources import files
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.opacity import OpaModit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure

config.update("jax_enable_x64", True)

testdata = {}
testdata["exomol"] = TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
testdata["hitemp"] = TESTDATA_CO_HITEMP_MODIT_EMISSION_REF


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_rt_modit(db, fig=False):
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    gravity = 2478.57
    mdb = mock_mdb(db)
    opa = OpaModit(
        mdb=mdb,
        nu_grid=nu_grid,
        Tarr_list=Tarr,
        Parr=art.pressure,
        dit_grid_resolution=0.2,
    )
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)
    F0 = art.run(dtau, Tarr)
    filename = files("exojax").joinpath("data/testdata/" + testdata[db])
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    maxres = np.max(residual)

    assert maxres < 3.0e-8  # 2.7443685102213067e-08 Feb. 17th 2025
