import pytest
import numpy as np
from exojax.spec.api import MdbHitemp


def test_moldb_hitemp():
    mdb = MdbHitemp(".database/CO/05_HITEMP2019/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)
    assert len(mdb.nu_lines) == 521

def test_moldb_hitemp_direct_name():
    mdb = MdbHitemp(".database/CO/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)
    assert len(mdb.nu_lines) == 521

def test_moldb_hitemp_direct_molecid():
    mdb = MdbHitemp(".database/05/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)
    assert len(mdb.nu_lines) == 521


def test_moldb_hitemp_interp():
    mdb = MdbHitemp(".database/CO/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)
    T = 1000.0
    assert mdb.qr_interp(1, T) == pytest.approx(3.5526853)
    qr = mdb.qr_interp_lines(T)
    ref = [3.5402815,3.5472596,3.5526853,3.5537634,3.5603657]
    uniq=np.unique(qr)
    assert uniq[0] == pytest.approx(ref[0])
    assert uniq[1] == pytest.approx(ref[1])
    assert uniq[2] == pytest.approx(ref[2])
    assert uniq[3] == pytest.approx(ref[3])
    assert uniq[4] == pytest.approx(ref[4])

def test_moldb_hitemp_isotope():
    num=[]
    for iso in [None,1,2,3,4,5,6]:
        mdb = MdbHitemp(".database/CO/",
                 nurange=[4200.0, 4300.0],
                 isotope=iso)
        num.append(len(mdb.nu_lines))
    num = np.array(num)
    assert num[0] == 4259
    assert num[0] == np.sum(num[1:])

if __name__ == "__main__":
    test_moldb_hitemp_isotope()
    #test_moldb_hitemp()
    #test_moldb_hitemp_direct_name()
    #test_moldb_hitemp_direct_molecid()
    #test_moldb_hitemp_interp()
    