""" unittest for initspec"""
import pytest
from exojax.spec.initspec import init_premodit
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.premodit import xsvector
from exojax.test.emulate_broadpar import mock_broadpar_exomol
from exojax.test.emulate_mdb import mock_mdbExoMol

def test_xsection_premodit():
    interval_contrast = 0.1
    dit_grid_resolution = 0.1
    Ttyp = 2000.0
    Ttest = 1500.0
    Ptest = 10.0
    
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
    print(ngamma_ref_grid[multi_index_uniqgrid[:,0]])
    xs = xsvector(Ttest, Ptest, lbd, R, pmarray, nu_grid, elower_grid, multi_index_uniqgrid,
             ngamma_ref_grid, n_Texp_grid, mdb.qr_interp)
    

if __name__ == "__main__":
    test_xsection_premodit()