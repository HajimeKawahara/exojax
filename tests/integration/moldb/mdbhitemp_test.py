import pytest
from exojax.spec.api import MdbHit

mdbCO=moldb.MdbHit('.database/CO/05_HITEMP2019/05_HITEMP2019.par.bz2',nus,crit=1.e-30)
def test_moldb_exomol():
    mdb = MdbExomol(".database/CO/12C-16O/Li2015",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30)


def test_moldb_exomol_interp():
    mdb = MdbExomol(".database/CO/12C-16O/Li2015",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30)
    T = 1000.0
    qr = mdb.qr_interp(T)
    print(qr)
    
if __name__ == "__main__":
    test_moldb_exomol()
    test_moldb_exomol_interp()
