"""CKD reflection must integrate the independent ordinate spectra."""

import numpy as np

from exojax.rt import ArtReflectPure


def test_ckd_reflection_matches_weighted_ordinate_spectra():
    art = ArtReflectPure(nlayer=4)
    dtau = np.linspace(0.01, 0.3, 24).reshape(4, 3, 2)
    weights = np.array([0.2, 0.3, 0.5])
    single_scattering_albedo = np.linspace(0.4, 0.8, 8).reshape(4, 2)
    asymmetry = np.linspace(0.2, 0.6, 8).reshape(4, 2)
    reflectivity = np.array([0.1, 0.3])
    incoming_flux = np.array([1.0, 2.0])
    arguments = (single_scattering_albedo, asymmetry, reflectivity, incoming_flux)
    expected = np.average(
        [art.run(dtau[:, ordinate, :], *arguments) for ordinate in range(3)],
        axis=0,
        weights=weights,
    )
    actual = art.run_ckd(dtau, *arguments, weights)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
