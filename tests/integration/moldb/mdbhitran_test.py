import pytest
import numpy as np
from exojax.database.api  import MdbHitran


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
    ref = [3.5402815, 3.5526853, 3.5537634, 3.5472596, 3.5603657]
    for i, isotope in enumerate(mdb.uniqiso):
        assert mdb.qr_interp(isotope, T) == pytest.approx(ref[i])
    
    assert mdb.qr_interp(1, T) == pytest.approx(3.5402815)
    
if __name__ == "__main__":
    test_moldb_hitran()
    test_moldb_hitran_direct_name()
    test_moldb_hitran_direct_molecid()
    test_moldb_hitran_interp()