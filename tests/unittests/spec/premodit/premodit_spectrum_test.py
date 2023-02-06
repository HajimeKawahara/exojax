""" short integration tests for PreMODIT spectrum"""
import pkg_resources
import jax.numpy as jnp
import pandas as pd
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec import rtransfer as rt

from exojax.spec.rtransfer import dtauM
from exojax.spec.rtransfer import rtrun
from exojax.spec.planck import piBarr
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.spec.opacalc import OpaPremodit
import pytest
from jax.config import config
from exojax.spec import api

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_rt_exomol(diffmode, fig=False):
    Parr, dParr, k = rt.pressure_layer(NP=100, numpy=True)
    T0_in = 1300.0
    alpha_in = 0.1
    Tarr = T0_in * (Parr)**alpha_in
    Tarr[Tarr < 400.0] = 400.0  #lower limit
    Tarr[Tarr > 1500.0] = 1500.0  #upper limit

    MMR = 0.1
    nu_grid, wav, res = wavenumber_grid(22900.0,
                                        23100.0,
                                        15000,
                                        unit='AA',
                                        xsmode="premodit")
    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',
    #                      nu_grid,
    #                      inherit_dataframe=False,
    #                      gpu_transfer=False)
    mdb = mock_mdbExomol()
    g = 2478.57
    #set OpaCalc
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[400, 1500.0],
                      dit_grid_resolution=0.1)

    print("dE=", opa.dE, "cm-1")
    xsm = opa.xsmatrix(Tarr, Parr)
    dtau = dtauM(dParr, jnp.abs(xsm), MMR * np.ones_like(Parr), mdb.molmass, g)
    sourcef = piBarr(Tarr, nu_grid)
    F0 = rtrun(dtau, sourcef)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    print(np.max(residual))
    assert np.all(residual < 0.01)
    return nu_grid, F0, dat["flux"].values


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_rt_hitemp(diffmode, fig=False):
    #mdb = mock_mdbHitemp(multi_isotope=False)
    
    Parr, dParr, k = rt.pressure_layer(NP=100, numpy=True)
    T0_in = 1300.0
    alpha_in = 0.1
    Tarr = T0_in * (Parr)**alpha_in
    Tarr[Tarr < 400.0] = 400.0  #lower limit
    Tarr[Tarr > 1500.0] = 1500.0  #upper limit

    MMR = 0.1
    nu_grid, wav, res = wavenumber_grid(22900.0,
                                        23100.0,
                                        15000,
                                        unit='AA',
                                        xsmode="premodit")
    #mdb = api.MdbHitemp('CO', nu_grid, gpu_transfer=False, isotope=1)
    mdb = mock_mdbHitemp()                                
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[400.0, 1500.0])
    g = 2478.57
    xsm = opa.xsmatrix(Tarr, Parr)
    dtau = dtauM(dParr, jnp.abs(xsm), MMR * np.ones_like(Parr), mdb.molmass, g)
    sourcef = piBarr(Tarr, nu_grid)
    F0 = rtrun(dtau, sourcef)

    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    print(np.max(residual))
    assert np.all(residual < 0.01)
    return nu_grid, F0, dat["flux"].values


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    diffmode = 0
    nus, F0, Fref = test_rt_exomol(diffmode)
    nus_hitemp, F0_hitemp, Fref_hitemp = test_rt_hitemp(diffmode)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    #ax.plot(nus, Fref, label="MODIT (ExoMol)")
    ax.plot(nus, F0, label="PreMODIT (ExoMol)", ls="dashed")
    plt.legend()
    #plt.yscale("log")
    ax = fig.add_subplot(312)
    #ax.plot(nus_hitemp, Fref_hitemp, label="MODIT (HITEMP)")
    ax.plot(nus_hitemp, F0_hitemp, label="PreMODIT (HITEMP)", ls="dashed")
    plt.legend()
    plt.ylabel("cross section (cm2)")

    ax = fig.add_subplot(313)
    ax.plot(nus,
            1.0 - F0 / Fref,
            alpha=0.7,
            label="dif = (MO - PreMO)/MO Exomol")
    ax.plot(nus_hitemp,
            1.0 - F0_hitemp / Fref_hitemp,
            alpha=0.7,
            label="dif = (MO - PreMO)/MO HITEMP")

    #plt.ylabel("dif")
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.05, color="gray", lw=0.5)
    plt.axhline(-0.05, color="gray", lw=0.5)
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.show()
