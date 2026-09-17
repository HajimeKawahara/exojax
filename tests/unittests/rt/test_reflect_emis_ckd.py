"""CKD reflection and emission must integrate independent ordinate spectra."""

import numpy as np
import pytest

from exojax.rt import ArtReflectEmis


@pytest.mark.parametrize("temperature_bottom", [1500.0, 2000.0])
def test_ckd_reflection_emission_matches_weighted_ordinate_spectra(temperature_bottom):
    nu_bands = np.array([1000.0, 2000.0])
    art = ArtReflectEmis(nlayer=4, nu_grid=nu_bands)
    dtau = np.linspace(0.01, 0.3, 24).reshape(4, 3, 2)
    weights = np.array([0.2, 0.3, 0.5])
    single_scattering_albedo = np.linspace(0.4, 0.7, 8).reshape(4, 2)
    asymmetry = np.linspace(0.2, 0.6, 8).reshape(4, 2)
    temperature = np.linspace(1000.0, temperature_bottom, 4)
    source_surface = np.array([1.0e-5, 2.0e-5])
    reflectivity = np.array([0.1, 0.3])
    incoming_flux = np.array([1.0, 2.0])
    arguments = (
        single_scattering_albedo,
        asymmetry,
        temperature,
        source_surface,
        reflectivity,
        incoming_flux,
    )
    expected = np.average(
        [art.run(dtau[:, ordinate, :], *arguments) for ordinate in range(3)],
        axis=0,
        weights=weights,
    )
    actual = art.run_ckd(dtau, *arguments, weights, nu_bands)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
