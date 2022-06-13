""" unittest for comparison of spectra calculated from mdb before and after plg"""
import numpy as np
from exojax.spec import plg
from exojax.spec import molinfo
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec import initspec
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_H2O_EXOMOL)
with open(filename, 'rb') as f:
    mdb = pickle.load(f)

def test_diff_frozen_vs_pseudo():
    wls, wll, nugrid_res = 15545, 15546, 0.05
    nus, wav, reso = gen_wavenumber_grid(wls, wll, int((wll-wls)/nugrid_res), unit="AA", xsmode="modit")
    cnu,indexnu,R,pmarray = initspec.init_modit(mdb.nu_lines,nus)

    Nelower = 7 
    Tgue = 3000. 
    errTgue = 500. 
    Mgue=41.
    Rgue=1.
    MMRgue=0.001
    molmass = molinfo.molmass("H2O")

    diff = plg.diff_frozen_vs_pseudo([0.8,], Tgue, errTgue, Mgue, Rgue, MMRgue, nus, mdb, molmass, Nelower)
    assert np.isclose(diff, 1872.8, rtol=1e-4)
    
if __name__ == "__main__":
    test_diff_frozen_vs_pseudo()
