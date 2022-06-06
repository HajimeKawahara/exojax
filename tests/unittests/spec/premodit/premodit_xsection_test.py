""" unittest for initspec"""
import pytest
import pkg_resources
import pandas as pd
import numpy as np
from exojax.spec.initspec import init_premodit
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.premodit import xsvector
from exojax.test.emulate_broadpar import mock_broadpar_exomol
from exojax.test.emulate_mdb import mock_mdbExoMol
from exojax.spec.molinfo import molmass
from exojax.spec import normalized_doppler_sigma
from exojax.test.data import TESTDATA_CO_EXOMOL_PREMODIT_SPECTRUM_REF


def test_xsection_premodit():
    interval_contrast = 0.1
    dit_grid_resolution = 0.1
    Ttyp = 2000.0
    Ttest = 1200.0
    Ptest = 1.0

    ngamma_ref, n_Texp = mock_broadpar_exomol()
    mdb = mock_mdbExoMol()
    #Mmol = molmass("CO")
    Nx = 5000
    
    nu_grid, wav, res = gen_wavenumber_grid(22800.0,
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
    xsv = xsvector(Ttest, Ptest, nsigmaD, lbd, R, pmarray, nu_grid, elower_grid,
                  multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/'+TESTDATA_CO_EXOMOL_PREMODIT_SPECTRUM_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))    
    return nu_grid,xsv

if __name__ == "__main__":
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_SPECTRUM_REF
    nus,xs = test_xsection_premodit()
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/'+TESTDATA_CO_EXOMOL_MODIT_SPECTRUM_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.title("premodit_xsection_test.py")
    ax.plot(nus,xs, label="PreMODIT")
    plt.legend()
    plt.yscale("log")
    plt.ylabel("cross section (cm2)")
    ax = fig.add_subplot(212)
    ax.plot(nus,1.0-xs/dat["xsv"],label="dif = (MODIT - PreMODIT)/MODIT")
    plt.ylabel("dif")
    plt.xlabel("wavenumber cm-1")
    plt.legend()
    plt.savefig("premodit.png")
    plt.show()
