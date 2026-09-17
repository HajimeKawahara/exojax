from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
import numpy as np
import pytest


@pytest.mark.parametrize(
    "make_mdb, reference_sum, high_temperature_sum",
    [
        (mock_mdbExomol, 3.260386610389642e-22, 1.2823972e-20),
        (mock_mdbHitemp, 3.2168443e-22, 1.2651083e-20),
    ],
    ids=["exomol", "hitemp"],
)
def test_line_strength_at_reference_and_high_temperature(
    make_mdb, reference_sum, high_temperature_sum
):
    mdb = make_mdb()
    reference = np.asarray(mdb.line_strength_ref_original)
    high_temperature = np.asarray(mdb.line_strength(1200.0))

    assert np.all(np.isfinite(reference))
    assert np.all(np.isfinite(high_temperature))
    assert np.sum(reference) == pytest.approx(reference_sum, rel=1.0e-6, abs=0.0)
    assert np.sum(high_temperature) == pytest.approx(
        high_temperature_sum, rel=1.0e-6, abs=0.0
    )
