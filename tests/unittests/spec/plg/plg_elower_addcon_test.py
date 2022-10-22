""" unittest for psudo line generator (plg)"""
import numpy as np
from exojax.spec import plg
from exojax.utils.grids import wavenumber_grid
from exojax.spec import initspec
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_H2O_EXOMOL)

def test_plg_elower_addcon():
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)

    wls, wll, wavenumber_grid_res = 15540, 15550, 0.05
    nus, wav, reso = wavenumber_grid(wls, wll, int((wll-wls)/wavenumber_grid_res), unit="AA", xsmode="modit")
    cnu,indexnu,R,pmarray = initspec.init_modit(mdb.nu_lines,nus)

    Nelower = 7 
    alpha_ref_grid, n_Texp_grid, index_gamma = plg.make_gamma_grid_exomol(mdb)
    Ngamma = len(alpha_ref_grid)

    qlogsij0, qcnu, num_unique, elower_grid, frozen_mask, nonzeropl_mask = plg.plg_elower_addcon(\
        index_gamma, Ngamma, cnu, indexnu, nus, mdb, 3000., 500., \
        Nelower=Nelower)    
    
    mdb, cnu, indexnu = plg.gather_lines(mdb, Ngamma, len(nus), Nelower, nus, cnu, indexnu, qlogsij0, \
        qcnu, elower_grid, alpha_ref_grid, n_Texp_grid, frozen_mask, nonzeropl_mask)

    assert mdb.A.shape[0] == 29
    assert np.sum(mdb.logsij0) == -3646.8888693415774
    assert np.sum(mdb.nu_lines) == 186549.27248329544
    assert np.sum(mdb.elower) == 469611.6255259078
    
if __name__ == "__main__":
    test_plg_elower_addcon()
