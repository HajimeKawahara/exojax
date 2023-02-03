from exojax.spec.api import _convert_proper_isotope
from exojax.spec.api import _isotope_index_from_isotope_number
from exojax.spec.api import _QT_interp
from exojax.spec.api import _qr_interp
from exojax.spec.api import _qr_interp_lines
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.utils.constants import Tref_original
import numpy as np
import pytest


def test__convert_proper_isotope():
    assert _convert_proper_isotope(0) is None
    assert _convert_proper_isotope(1) == "1"
    assert _convert_proper_isotope(None) is None


def test__isotope_index_from_isotope_number():
    uniqiso = np.array([1, 2, 3, 5, 6])
    assert _isotope_index_from_isotope_number(1, uniqiso) == 0
    assert _isotope_index_from_isotope_number(6, uniqiso) == 4


def test__QT_interp():
    mdb = mock_mdbHitemp(multi_isotope=True)
    T = 1000.0
    isotope_index = _isotope_index_from_isotope_number(1, mdb.uniqiso)
    QT = _QT_interp(isotope_index, T, mdb.T_gQT, mdb.gQT)
    assert QT == pytest.approx(380.297)


def test__qr_interp():
    mdb = mock_mdbHitemp(multi_isotope=True)
    T = 1000.0
    isotope_index = _isotope_index_from_isotope_number(1, mdb.uniqiso)
    qr = _qr_interp(isotope_index, T, mdb.T_gQT, mdb.gQT, Tref_original)
    assert qr == pytest.approx(3.5402815)


def test__qr_interp_lines():
    mdb = mock_mdbHitemp(multi_isotope=True)
    T = 1000.0
    val = np.sum(
        _qr_interp_lines(T, mdb.isoid, mdb.uniqiso, mdb.T_gQT, mdb.gQT,
                         Tref_original))

    assert val == pytest.approx(141.61127)


def test__exact_isotope_name():
    mdb = mock_mdbHitemp(multi_isotope=True)
    assert mdb.exact_isotope_name(1) == "(12C)(16O)"


def test_molmass():
    mdb = mock_mdbHitemp(multi_isotope=True)
    print(mdb.molmass)
    #assert mdb.exact_isotope_name(1) == "(12C)(16O)"


if __name__ == "__main__":
    #    test__convert_proper_isotope()
    #    test__isotope_index_from_isotope_number()
    #    test__QT_interp()
    #    test__qr_interp()
    #    test__qr_interp_lines()
    test_molmass()