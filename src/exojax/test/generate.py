"""generate test data

"""
from exojax.spec import moldb
from exojax.spec.setrt import gen_wavenumber_grid
import pickle
import numpy as np

def gendata_moldb_exomol():
    """generate test data for CO exomol
    """
    from exojax.test.data import TESTDATA_moldb_CO_EXOMOL as filename
    Nx=10000
    nus, wav, res = gen_wavenumber_grid(22920.0,24000.0, Nx, unit='AA')
    mdbCO = moldb.MdbExomol('.database/CO/12C-16O/Li2015', nus, crit=1e-35, Ttyp=296.0)
    with open(filename, 'wb') as f:
        pickle.dump(mdbCO, f)

        
if __name__ == "__main__":
    gendata_moldb_exomol()
