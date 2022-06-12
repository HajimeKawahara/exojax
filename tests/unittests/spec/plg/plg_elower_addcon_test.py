""" unittest for psudo line generator (plg)"""
import numpy as np
from exojax.spec import plg
from exojax.spec.rtransfer import nugrid
from exojax.spec import initspec
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_H2O_EXOMOL)

def test_plg_elower_addcon():
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)

    wls, wll, nugrid_res = 15540, 15550, 0.05
    nus, wav, reso = nugrid(wls, wll, int((wll-wls)/nugrid_res), unit="AA", xsmode="modit")
    cnu,indexnu,R,pmarray = initspec.init_modit(mdb.nu_lines,nus)

    Nelower = 7 
    Tgue = 3000. 
    errTgue = 500. 
    alpha_ref_grid, n_Texp_grid, index_gamma = plg.make_gamma_grid_exomol(mdb)
    Ngamma = len(alpha_ref_grid)

    qlogsij0, qcnu, num_unique, elower_grid, frozen_mask, nonzeropl_mask = plg.plg_elower_addcon(\
        index_gamma, Ngamma, cnu, indexnu, nus, mdb, Tgue, errTgue, \
        Nelower=Nelower, Tmargin=0., coefTgue=1.)    
    
    mdb, cnu, indexnu = plg.gather_lines(mdb, Ngamma, len(nus), Nelower, nus, cnu, indexnu, qlogsij0, \
        qcnu, elower_grid, alpha_ref_grid, n_Texp_grid, frozen_mask, nonzeropl_mask)

    assert mdb.A.shape[0] == 1317
    assert np.sum(mdb.logsij0) == -169998.69144105286
    assert np.sum(mdb.nu_lines) == 8471903.657835286
    assert np.sum(mdb.elower) == 22127273.078465365
    
if __name__ == "__main__":
    test_plg_elower_addcon()
