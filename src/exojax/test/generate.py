"""generate test data

"""
from exojax.spec import moldb
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
from exojax.spec.setrt import gen_wavenumber_grid

import pickle

def gendata_exomol():
    from exojax.test.data import TESTDATA_moldb_CO as filename
    Nx=1500
    nus, wav, res = gen_wavenumber_grid(22920.0,24000.0, Nx, unit='AA')
    mdb = moldb.MdbExomol('.database/CO/12C-16O/Li2015', nus, crit=1e-35, Ttyp=296.0)
    with open(filename, 'wb') as f:
        pickle.dump(mdb, f)
        
if __name__ == "__main__":
    gendata_exomol()
