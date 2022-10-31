""" short integration tests for PreMODIT cross section"""
import pytest
import pkg_resources
import pandas as pd
import numpy as np
from exojax.spec.initspec import init_premodit
from exojax.utils.grids import wavenumber_grid
from exojax.spec.premodit import xsvector
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.molinfo import molmass
from exojax.spec import normalized_doppler_sigma
from exojax.test.data import TESTDATA_CO_EXOMOL_PREMODIT_XS_REF
from exojax.test.data import TESTDATA_CO_HITEMP_PREMODIT_XS_REF
import warnings


def test_xsection_premodit_exomol():
    interval_contrast = 0.1
    dit_grid_resolution = 0.1
    Ttyp = 2000.0
    Ttest = 1200.0
    Ptest = 1.0
    mdb = mock_mdbExomol()
    #Mmol = molmass("CO")
    Nx = 5000

    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")

    lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray = init_premodit(
        mdb.nu_lines,
        nu_grid,
        mdb.elower,
        mdb.alpha_ref,
        mdb.n_Texp,
        mdb.Sij0,
        Ttyp,
        interval_contrast=interval_contrast,
        dit_grid_resolution=dit_grid_resolution,
        warning=False)

    Mmol = molmass("CO")
    nsigmaD = normalized_doppler_sigma(Ttest, Mmol, R)
    qt = mdb.qr_interp(Ttest)
    xsv = xsvector(Ttest, Ptest, nsigmaD, lbd, R, pmarray, nu_grid,
                   elower_grid, multi_index_uniqgrid, ngamma_ref_grid,
                   n_Texp_grid, qt)

    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_PREMODIT_XS_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))
    return nu_grid, xsv


def test_xsection_premodit_hitemp():
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

    lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray = init_premodit(
        mdb.nu_lines,
        nu_grid,
        mdb.elower,
        mdb.gamma_air,
        mdb.n_air,
        mdb.Sij0,
        Ttyp,
        interval_contrast=interval_contrast,
        dit_grid_resolution=dit_grid_resolution,
        warning=False)

    Mmol = molmass("CO")
    nsigmaD = normalized_doppler_sigma(Ttest, Mmol, R)
    qt = mdb.qr_interp(1, Ttest)
    message = "Here, we use a single partition function qt for isotope=1 despite of several isotopes."
    warnings.warn(message, UserWarning)
    xsv = xsvector(Ttest, Ptest, nsigmaD, lbd, R, pmarray, nu_grid,
                   elower_grid, multi_index_uniqgrid, ngamma_ref_grid,
                   n_Texp_grid, qt)
    import matplotlib.pyplot as plt
    plt.plot(nu_grid, xsv)
    plt.yscale("log")
    plt.show()
    #filename = pkg_resources.resource_filename(
    #    'exojax', 'data/testdata/'+TESTDATA_CO_HITEMP_PREMODIT_XS_REF)
    #dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    #assert np.all(xsv == pytest.approx(dat["xsv"].values))
    return nu_grid, xsv


if __name__ == "__main__":
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_XS_REF
    from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_XS_REF
    import matplotlib.pyplot as plt
    #import jax.profiler

    database = "hitemp"
    #database = "exomol"
    if database == "exomol":
        nus, xs = test_xsection_premodit_exomol()
        filename = pkg_resources.resource_filename(
            'exojax', 'data/testdata/' + TESTDATA_CO_EXOMOL_MODIT_XS_REF)
    elif database == "hitemp":
        nus, xs = test_xsection_premodit_hitemp()
        filename = pkg_resources.resource_filename(
            'exojax', 'data/testdata/' + TESTDATA_CO_HITEMP_MODIT_XS_REF)

    #xs.block_until_ready()
    #jax.profiler.save_device_memory_profile("memory.prof")

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.title("premodit_xsection_test.py")
    ax.plot(nus, xs, label="PreMODIT")
    plt.legend()
    plt.yscale("log")
    plt.ylabel("cross section (cm2)")
    ax = fig.add_subplot(212)
    ax.plot(nus, 1.0 - xs / dat["xsv"], label="dif = (MODIT - PreMODIT)/MODIT")
    plt.ylabel("dif")
    plt.xlabel("wavenumber cm-1")
    plt.legend()
    plt.savefig("premodit.png")
    plt.show()
