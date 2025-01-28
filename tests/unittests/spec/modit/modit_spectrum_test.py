""" short integration tests for MODIT spectrum"""

import pytest
from importlib.resources import files
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.spec.opacalc import OpaModit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.atmrt import ArtEmisPure

config.update("jax_enable_x64", True)

testdata = {}
testdata["exomol"] = TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
testdata["hitemp"] = TESTDATA_CO_HITEMP_MODIT_EMISSION_REF


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_rt(db, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.1)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
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
    print(np.max(residual))
    # assert np.all(residual < 1.e-6)
    return nu_grid, F0, dat["flux"].values


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    diffmode = 0
    nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)  #
    nus, F0, Fref = test_rt("exomol", diffmode)  #
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(nus, Fref, label="MODIT (ExoMol)")
    ax.plot(nus, F0, label="MODIT (ExoMol, new)", ls="dashed")
    plt.legend()
    # plt.yscale("log")
    ax = fig.add_subplot(312)
    ax.plot(nus_hitemp, Fref_hitemp, label="MODIT (HITEMP)")
    ax.plot(nus_hitemp, F0_hitemp, label="MODIT (HITEMP, new)", ls="dashed")
    plt.legend()
    plt.ylabel("flux (cgs)")

    ax = fig.add_subplot(313)
    ax.plot(nus, 1.0 - F0 / Fref, alpha=0.7, label="dif (Exomol)")
    ax.plot(nus_hitemp, 1.0 - F0_hitemp / Fref_hitemp, alpha=0.7, label="dif (HITEMP)")
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.05, color="gray", lw=0.5)
    plt.axhline(-0.05, color="gray", lw=0.5)
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.show()
