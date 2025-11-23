import pytest
from exojax.database.exomol.api import MdbExomol
from exojax.utils.constants import Tref_original

def test_moldb_exomol():
    mdb = MdbExomol(".database/CO/12C-16O/Li2015",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30)


def test_moldb_exomol_interp():
    mdb = MdbExomol(".database/CO/12C-16O/Li2015",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30)
    T = 1000.0
    qr = mdb.qr_interp(T,Tref_original)
    ref = 3.5402875
    
    assert qr == pytest.approx(ref)
    
if __name__ == "__main__":
    test_moldb_exomol()
    test_moldb_exomol_interp()