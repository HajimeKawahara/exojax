""" short integration tests for PreMODIT cross section"""
import pytest
import pkg_resources
import pandas as pd
import numpy as np
from exojax.spec.initspec import init_premodit
from exojax.spec.opacalc import OpaPremodit
from exojax.utils.grids import wavenumber_grid
from exojax.spec.premodit import xsvector_second, xsvector_first, xsvector_zeroth
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.molinfo import molmass
from exojax.spec import normalized_doppler_sigma
from exojax.test.data import TESTDATA_CO_EXOMOL_PREMODIT_XS_REF
from exojax.test.data import TESTDATA_CO_HITEMP_PREMODIT_XS_REF
import warnings


@pytest.mark.parametrize("diffmode", [0, 1, 2])
def test_xsection_premodit_exomol(diffmode):
    #interval_contrast = 0.3
    Tref = 500.0
    Twt = 1000.0
    Ttest = 1200.0 #fix to compare w/ precomputed xs by MODIT.
    Ptest = 1.0
    mdb = mock_mdbExomol()
    mdb.change_reference_temperature(Tref)
    #Mmol = molmass("CO")
    Nx = 5000
    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")

    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, Twt=Twt)
    lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    
    #Mmol = molmass("CO")
    Mmol = mdb.molecular_mass
    nsigmaD = normalized_doppler_sigma(Ttest, Mmol, R)
    qt = mdb.qr_interp(Ttest)
    if diffmode == 0:
        xsv = xsvector_zeroth(Ttest, Ptest, nsigmaD, lbd_coeff, Tref, R,
                              pmarray, opa.nu_grid, elower_grid,
                              multi_index_uniqgrid, ngamma_ref_grid,
                              n_Texp_grid, qt)
    elif diffmode == 1:
        xsv = xsvector_first(Ttest, Ptest, nsigmaD, lbd_coeff, Tref, Twt,
                       R, pmarray, opa.nu_grid, elower_grid,
                       multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt)
    elif diffmode == 2:
        xsv = xsvector_second(Ttest, Ptest, nsigmaD, lbd_coeff, Tref, Twt,
                       R, pmarray, opa.nu_grid, elower_grid,
                       multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt)

    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_PREMODIT_XS_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    #assert np.all(xsv == pytest.approx(dat["xsv"].values))
    return opa.nu_grid, xsv, opa.dE, Twt, Tref, Ttest


@pytest.mark.parametrize("diffmode", [0, 1])
def test_xsection_premodit_hitemp(diffmode):
    interval_contrast = 0.1
    dit_grid_resolution = 0.1
    Ttyp = 2000.0
    Ttest = 1200.0
    Ptest = 1.0
    mdb = mock_mdbHitemp(multi_isotope=False)
    #Mmol = molmass("CO")
    Nx = 5000

    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")

    lbd_zeroth, lbd_first, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray = init_premodit(
        mdb.nu_lines,
        nu_grid,
        mdb.elower,
        mdb.gamma_air,
        mdb.n_air,
        mdb.Sij0,
        Ttyp,
        interval_contrast=interval_contrast,
        dit_grid_resolution=dit_grid_resolution,
        diffmode=diffmode,
        warning=False)

    Mmol = molmass("CO")
    nsigmaD = normalized_doppler_sigma(Ttest, Mmol, R)
    qt = mdb.qr_interp(1, Ttest)
    message = "Here, we use a single partition function qt for isotope=1 despite of several isotopes."
    warnings.warn(message, UserWarning)
    if diffmode == 0:
        xsv = xsvector_zeroth(Ttest, Ptest, nsigmaD, lbd_zeroth, R, pmarray,
                              nu_grid, elower_grid, multi_index_uniqgrid,
                              ngamma_ref_grid, n_Texp_grid, qt)
    elif diffmode == 1:
        xsv = xsvector(Ttest, Ptest, nsigmaD, lbd_zeroth, lbd_first, R,
                       pmarray, nu_grid, elower_grid, multi_index_uniqgrid,
                       ngamma_ref_grid, n_Texp_grid, qt)

    #np.savetxt(TESTDATA_CO_HITEMP_PREMODIT_XS_REF,np.array([nu_grid,xsv]).T,delimiter=",")
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_HITEMP_PREMODIT_XS_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))
    return nu_grid, xsv


if __name__ == "__main__":
    #comparison with MODIT
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_XS_REF
    import matplotlib.pyplot as plt
    #import jax.profiler
    diffmode = 2
    nus, xs, dE, Twt, Tref, Tin = test_xsection_premodit_exomol(diffmode)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_MODIT_XS_REF)

    #xs.block_until_ready()
    #jax.profiler.save_device_memory_profile("memory.prof")

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    #plt.title("premodit_xsection_test.py diffmode=" + str(diffmode))
    plt.title("diffmode=" + str(diffmode) + " T=" + str(Tin) + " Tref=" +
              str(Tref) + " Twt=" + str(Twt) + " dE=" + str(dE))
    ax.plot(nus, xs, label="PreMODIT")
    ax.plot(nus, dat["xsv"], label="MODIT")
    plt.legend()
    plt.yscale("log")
    plt.ylabel("cross section (cm2)")
    ax = fig.add_subplot(212)
    ax.plot(nus, 1.0 - xs / dat["xsv"], label="dif = (MODIT - PreMODIT)/MODIT")
    plt.ylabel("dif")
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    plt.ylim(-0.03, 0.03)
    plt.legend()
    plt.savefig("premodit" + str(diffmode) + ".png")
    plt.show()
