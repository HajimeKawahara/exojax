from exojax.test.emulate_mdb import mock_mdbHargreaves
from exojax.database.hargreaves.api import _set_wavenum_hargreaves
import numpy as np
import pytest

@pytest.mark.parametrize(
    "input_nurange, expected_output, expect_warning",
    [
        ([15820.0, 20040.0], (15820.0, 20040.0), False),
        (None, (0.0, 0.0), True),
        ([-np.inf, 20040.0], (None, 20040.0), False),
        ([15820., np.inf], (15820., None), False),
        ([-np.inf, np.inf], (None, None), False),
    ]
)
def test_set_wavenum(input_nurange, expected_output, expect_warning):
    if expect_warning:
        with pytest.warns(UserWarning, match="nurange=None."):
            result = _set_wavenum_hargreaves(input_nurange)
    else:
        result = _set_wavenum_hargreaves(input_nurange)
    assert result == expected_output

def test_line_strength_hargreaves():
    mdb = mock_mdbHargreaves()
    df = mdb.df
    Sij0 = df["Sij0"].values
    assert np.sum(Sij0) == pytest.approx(1.59740e-17)

if __name__ == "__main__": 
    test_line_strength_hargreaves()
