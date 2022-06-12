""" unittest for preparation of gamma grid fr plg"""
import numpy as np
from exojax.spec import plg
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_H2O_EXOMOL)

def test_make_gamma_grid_exomol_plg():
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    alpha_ref_grid, n_Texp_grid, index_gamma = plg.make_gamma_grid_exomol(mdb)

    assert len(alpha_ref_grid) == 100
    assert np.isclose(np.sum(alpha_ref_grid), 3.7375998)
    assert np.isclose(np.sum(n_Texp_grid), 24.8)
    assert np.sum(index_gamma) == 123958
    
if __name__ == "__main__":
    test_make_gamma_grid_exomol_plg()
