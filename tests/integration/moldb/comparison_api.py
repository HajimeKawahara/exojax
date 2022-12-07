"""comparison the new api with the old moldb
"""
import pytest
import numpy as np
from exojax.spec.api import MdbExomol as MdbExomol_api
from exojax.spec.moldb import MdbExomol as MdbExomol_orig
from exojax.spec.api import MdbHitemp as MdbHit_api
from exojax.spec.moldb import MdbHit as MdbHit_orig


def comparison_moldb_exomol():
    crit = 1.e-30
    morig = MdbExomol_orig(".database_/CO/12C-16O/Li2015",
                           nurange=[4200.0, 4300.0],
                           crit=crit,
                           margin=0.0)
    mapi = MdbExomol_api(".database/CO/12C-16O/Li2015",
                         nurange=[4200.0, 4300.0],
                         crit=crit,
                         margin=0.0)
    assert np.all(mapi.A - morig.A) == 0.0

    T = 1000.0
    qr_api = mapi.qr_interp(T)
    qr_orig = morig.qr_interp(T)
    assert qr_api - qr_orig == pytest.approx(0.0)


def comparison_moldb_hitemp():
    crit = 1.e-30
    morig = MdbHit_orig(".database_/CO/05_HITEMP2019/05_HITEMP2019.par.bz2",
                        nurange=[4200.0, 4300.0],
                        crit=crit)
    mapi = MdbHit_api(".database/CO/",
                      nurange=[4200.0, 4300.0],
                      crit=crit)
    assert np.all(mapi.A - morig.A) == 0.0

    T = 1000.0
    qr_api = mapi.qr_interp(1, T)
    qr_orig = morig.qr_interp(1, T)
    assert qr_api - qr_orig == pytest.approx(0.0)

    qr_lines_api = mapi.qr_interp_lines(T)
    qr_lines_orig = morig.qr_interp_lines(T)
    assert np.all(qr_api - qr_orig) == pytest.approx(0.0)
        
if __name__ == "__main__":
    comparison_moldb_hitemp()
    comparison_moldb_exomol()
