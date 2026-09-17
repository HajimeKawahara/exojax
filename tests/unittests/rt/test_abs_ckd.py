"""Check band transmission against Beer-Lambert attenuation."""

import numpy as np
import pytest

from exojax.rt import ArtAbsPure


@pytest.mark.parametrize("mu_out", [None, 0.8], ids=["ground", "top"])
def test_ckd_absorption_matches_beer_lambert(mu_out):
    art = ArtAbsPure(pressure_top=0.1, pressure_btm=10.0, nlayer=4)
    dtau = np.linspace(0.01, 0.3, 24).reshape(4, 3, 2)
    weights = np.array([0.2, 0.3, 0.5])
    incoming_flux = np.array([1.0, 2.0])
    mu_in = 0.5
    path_factor = 1.0 / mu_in + (0.0 if mu_out is None else 1.0 / mu_out)
    expected = incoming_flux * np.average(
        np.exp(-path_factor * dtau.sum(axis=0)), axis=0, weights=weights
    )
    actual = art.run_ckd(
        dtau, art.pressure_boundary[-1], incoming_flux, mu_in, mu_out, weights
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
