import pytest
import numpy as np
from exojax.spec.api import MdbHitran


def test_moldb_hitran():
    mdb = MdbHitran(".database/CO/05_hit12.par",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)    
    assert len(mdb.nu_lines) == 222

def test_moldb_hitran_direct_name():
    mdb = MdbHitran(".database/CO/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)    
    assert len(mdb.nu_lines) == 222

def test_moldb_hitran_direct_molecid():
    mdb = MdbHitran(".database/05/",
                 nurange=[4200.0, 4300.0],
                 crit=1.e-30)    
    assert len(mdb.nu_lines) == 222


def test_moldb_hitran_interp():
    mdb = MdbHitran(".database/CO/05_hit12.par",
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


if __name__ == "__main__":
    test_moldb_hitran()
    test_moldb_hitran_direct_name()
    test_moldb_hitran_direct_molecid()
    test_moldb_hitran_interp()