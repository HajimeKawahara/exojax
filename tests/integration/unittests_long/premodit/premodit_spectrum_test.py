""" short integration tests for PreMODIT spectrum"""

import pytest
from importlib.resources import files
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "db, diffmode", [("exomol", 1), ("exomol", 2), ("hitemp", 1), ("hitemp", 2)]
)
def test_rt(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb =MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[art.Tlow, art.Thigh]
    )

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

    F0 = art.run(dtau, Tarr)

    if db == "hitemp":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
        )       
    elif db == "exomol":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
        )
        
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    print(np.max(residual))
    # assert np.all(residual < 0.007)
    return nu_grid, F0, dat["flux"].values


@pytest.mark.parametrize(
    "db, diffmode",
    [
        ("exomol", 0),
        ("exomol", 1),
        ("exomol", 2),
        ("hitemp", 0),
        ("hitemp", 1),
        ("hitemp", 2),
    ],
)
def test_rt_for_single_broadening_parameters(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb =MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=diffmode,
        auto_trange=[art.Tlow, art.Thigh],
        broadening_resolution={"mode": "single", "value": None},
    )
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)
    F0 = art.run(dtau, Tarr)

    if db == "hitemp":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
        )        
    elif db == "exomol":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
        )

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    print(np.max(residual))
    # assert np.all(residual < 0.05)
    return nu_grid, F0, dat["flux"].values


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db = "hitemp"
    diffmode = 0
    nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)
    nus, F0, Fref = test_rt("exomol", diffmode)  #

    # nus_hitemp, F0_hitemp, Fref_hitemp = test_rt_for_single_broadening_parameters(
    #    "hitemp", diffmode)
    # nus, F0, Fref = test_rt_for_single_broadening_parameters(
    #    "exomol", diffmode)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    # ax.plot(nus, Fref, label="MODIT (ExoMol)")
    ax.plot(nus, F0, label="PreMODIT (ExoMol)", ls="dashed")
    plt.legend()
    # plt.yscale("log")
    ax = fig.add_subplot(312)
    # ax.plot(nus_hitemp, Fref_hitemp, label="MODIT (HITEMP)")
    ax.plot(nus_hitemp, F0_hitemp, label="PreMODIT (HITEMP)", ls="dashed")
    plt.legend()
    plt.ylabel("flux (cgs)")

    ax = fig.add_subplot(313)
    ax.plot(nus, 1.0 - F0 / Fref, alpha=0.7, label="dif = (MO - PreMO)/MO Exomol")
    ax.plot(
        nus_hitemp,
        1.0 - F0_hitemp / Fref_hitemp,
        alpha=0.7,
        label="dif = (MO - PreMO)/MO HITEMP",
    )
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.05, color="gray", lw=0.5)
    plt.axhline(-0.05, color="gray", lw=0.5)
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    # plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.show()
