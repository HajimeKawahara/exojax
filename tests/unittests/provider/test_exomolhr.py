from exojax.test.data import TESTDATA_EXOMOLHR_CSV
from exojax.test.data import get_testdata_filename
from exojax.provider.exomolhr import _load_exomolhr_csv
import pytest

def test_load_exomolhr_csv_preserves_quoted_headers_and_values():
    csv_path = get_testdata_filename(TESTDATA_EXOMOLHR_CSV)
    df = _load_exomolhr_csv(csv_path)

    assert df.shape == (195, 21)
    assert df.loc[0, 'E"'] == pytest.approx(3766.378879)
    assert df.loc[0, "Grve'"] == "A2"
    assert df.loc[0, "S"] == pytest.approx(1.270814109606162e-29, rel=1.0e-12, abs=0.0)
