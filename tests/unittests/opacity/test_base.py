"""Test shared opacity grid and aliasing behavior."""

import numpy as np
import pytest

from exojax.opacity.base import OpaCalc


def test_opacalc_rejects_invalid_alias():
    opa = OpaCalc(np.geomspace(1900.0, 2300.0, 16))
    opa.alias = "invalid"
    with pytest.raises(ValueError, match="alias should be"):
        opa.set_aliasing()


@pytest.mark.parametrize("cutwing,expected_length", [(1.0, 16), (0.5, 8)])
def test_opacalc_filter_length_from_cutwing(cutwing, expected_length):
    opa = OpaCalc(np.geomspace(1900.0, 2300.0, 16))
    opa.cutwing = cutwing
    opa.set_filter_length_oneside_from_cutwing()
    assert opa.filter_length_oneside == expected_length
