from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
import numpy as np
import pytest


def test_line_strength_exomol():
    mdb = mock_mdbExomol()
    assert pytest.approx(np.sum(mdb.line_strength_ref)) == 3.260386610389642e-22


def test_line_strength_exomol_t():
    mdb = mock_mdbExomol()
    mdb.change_reference_temperature(1200.0)
    mask = np.isfinite(mdb.line_strength_ref)
    val = np.sum(mdb.line_strength_ref[mask])
    assert val == 1.2823972e-20


def test_line_strength_hitemp():
    mdb = mock_mdbHitemp()
    assert pytest.approx(np.sum(mdb.line_strength_ref)) == 3.2168443e-22


def test_line_strength_hitemp_t():
    mdb = mock_mdbHitemp()
    mdb.change_reference_temperature(1200.0)
    mask = np.isfinite(mdb.line_strength_ref)
    val = np.sum(mdb.line_strength_ref[mask])
    assert val == 1.2651083e-20


if __name__ == "__main__":
    # test_line_strength_exomol_t()
    test_line_strength_hitemp_t()
