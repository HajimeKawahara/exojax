import pytest
import numpy as np
from exojax.database.hitemp.api import MdbHitemp


def test_moldb_hitemp():
    mdb = MdbHitemp(".database/CO/05_HITEMP2019/",
                    nurange=[4200.0, 4300.0],
                    crit=1.e-30,
                    isotope=None,
                    inherit_dataframe=True)
    assert len(mdb.nu_lines) == 521


def test_moldb_hitemp_direct_name():
    mdb = MdbHitemp(".database/CO/",
                    nurange=[4200.0, 4300.0],
                    isotope=None,
                    crit=1.e-30)
    assert len(mdb.nu_lines) == 521


def test_moldb_hitemp_direct_molecid():
    mdb = MdbHitemp(".database/05/",
                    nurange=[4200.0, 4300.0],
                    isotope=None,
                    crit=1.e-30)
    assert len(mdb.nu_lines) == 521


def test_moldb_hitemp_interp():
    from exojax.utils.constants import Tref_original
    mdb = MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0], crit=1.e-30)
    T = 1000.0
    ref = [3.5402815, 3.5526853, 3.5537634, 3.5472596, 3.5603657]
    for i, isotope in enumerate(mdb.uniqiso):
        assert mdb.qr_interp(isotope, T, Tref_original) == pytest.approx(ref[i])

    qr = mdb.qr_interp_lines(T, Tref_original)


def test_moldb_hitemp_isotope():
    num = []
    for iso in [0, 1, 2, 3, 4, 5, 6]:
        mdb = MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0], isotope=iso)
        num.append(len(mdb.nu_lines))
    num = np.array(num)
    assert num[0] == 4259
    assert num[0] == np.sum(num[1:])
    mdb = MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0], isotope=None)
    assert len(mdb.nu_lines) == np.sum(num[1:])


if __name__ == "__main__":
    #test_moldb_hitemp_isotope()
    test_moldb_hitemp()
    #test_moldb_hitemp_direct_name()
    #test_moldb_hitemp_direct_molecid()
    #test_moldb_hitemp_interp()
