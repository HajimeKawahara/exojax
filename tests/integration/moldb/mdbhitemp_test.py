import pytest
from exojax.spec.api import MdbHit


def test_moldb_hitemp():
    mdb = MdbHit(".database/CO/05_HITEMP2019/05_HITEMP2019.par.bz2",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30)
    print(mdb.fetch_url_Nlines_wmin_wmax())
    print(mdb.fetch_urlnames())
    print(mdb)
#def test_moldb_hitemp_interp():
#    mdb = MdbHit(".database/CO/12C-16O/Li2015",
#                    nurange=[4200.0, 4300.0],
#                    crit=1.e-30)
 #   T = 1000.0
 #   qr = mdb.qr_interp(T)
 #   print(qr)
    
if __name__ == "__main__":
    test_moldb_hitemp()
 #   test_moldb_hitemp_interp()