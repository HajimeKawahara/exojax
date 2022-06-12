""" unittest for optimization of 'coefTgue' for plg"""
import numpy as np
from exojax.spec import plg
from exojax.spec import molinfo
from exojax.spec.rtransfer import nugrid
from exojax.spec import initspec
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_H2O_EXOMOL)

def test_optimize_coefTgue():
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    molmass = molinfo.molmass("H2O")

    wls, wll, nugrid_res = 15545, 15546, 0.05
    nus, wav, reso = nugrid(wls, wll, int((wll-wls)/nugrid_res), unit="AA", xsmode="modit")
    cnu,indexnu,R,pmarray = initspec.init_modit(mdb.nu_lines,nus)

    Nelower = 7 
    Tgue = 3000. 
    errTgue = 500. 

    coefTgue = plg.optimize_coefTgue(Tgue, nus, mdb, molmass, Nelower, errTgue)

    assert np.isclose(coefTgue, 0.6737, rtol=1e-4)
    
if __name__ == "__main__":
    test_optimize_coefTgue()
