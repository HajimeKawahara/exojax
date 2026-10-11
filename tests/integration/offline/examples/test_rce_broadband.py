"""Offline energy and boundary checks for broadband RCE source integration."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.ckd.mixing import mix_ckd_rorr
from exojax.rt.planck import piBarr


@pytest.fixture
def broadband(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("_rce_broadband")


def _bands(lower=1.0, upper=300000.0, count=48):
    edges = np.geomspace(lower, upper, count + 1)
    return np.column_stack((edges[:-1], edges[1:]))


def _angles():
    mu, weights = np.polynomial.legendre.leggauss(4)
    return jnp.asarray((mu + 1.0) / 2.0), jnp.asarray(weights / 2.0)


def test_planck_band_integration_converges_to_stefan_boltzmann(broadband):
    temperatures = jnp.array([1500.0, 3000.0, 6400.0])
    target = 5.670374419e-5 * np.asarray(temperatures)**4
    fluxes = []
    for edges in (_bands(1000.0, 20000.0), _bands()):
        nodes, weights = broadband.make_band_quadrature(edges)
        source = broadband.band_planck_flux(temperatures, nodes, weights)
        fluxes.append(np.sum(source * np.diff(edges, axis=1).ravel(), axis=-1))
    assert np.all(np.abs(fluxes[1] - target) < np.abs(fluxes[0] - target))
    # ExoJAX's historical Planck constants differ by 1.34 ppm from CODATA 2018.
    np.testing.assert_allclose(fluxes[1], target, rtol=1.5e-6)
    assert broadband.band_planck_flux(3000.0, nodes, weights).shape == (48,)


def test_stellar_average_preserves_narrow_features_and_requires_coverage(broadband):
    nu = np.array([1.0, 2.0, 2.01, 2.02, 3.0, 4.0])
    flux = np.array([1.0, 1.0, 101.0, 1.0, 1.0, 1.0])
    edges = np.array([[1.0, 2.015], [2.015, 4.0]])
    average = broadband.band_stellar_flux(nu, flux, edges)
    np.testing.assert_allclose(np.sum(average * np.diff(edges).ravel()), 4.0)
    np.testing.assert_allclose(
        average * np.diff(edges).ravel(), [1.015 + 0.875, 1.985 + 0.125],
    )
    with pytest.raises(ValueError, match="cover all band edges"):
        broadband.band_stellar_flux(nu, flux, [[0.5, 4.0]])
    with pytest.raises(ValueError, match="increasing wavenumbers"):
        broadband.band_stellar_flux(nu[::-1], flux[::-1], edges)
    with pytest.raises(ValueError, match="nonoverlapping"):
        broadband.make_band_quadrature([[1.0, 3.0], [2.0, 4.0]])


def test_transparent_column_preserves_blackbody_and_horizontal_irradiation(broadband):
    edges = _bands()
    nodes, band_weights = broadband.make_band_quadrature(edges)
    widths = jnp.asarray(np.diff(edges).ravel())
    boundaries = jnp.array([0.001, 0.1, 1.0])
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = jnp.array([1800.0, 2300.0])
    mus, angular_weights = _angles()
    g_weights = jnp.array([0.3, 0.7])
    star = broadband.band_planck_flux(6400.0, nodes, band_weights)
    incident = broadband.irradiation_flux(star, a_over_rstar=3.5, amplitude=0.4)
    for mu0 in (0.2, 0.8):
        spectral = broadband.ckd_absorption_fluxes(
            jnp.zeros((2, 2, len(edges))), temperature, 2500.0,
            pressure, boundaries, nodes, band_weights, mus, angular_weights,
            incident, mu0,
        )
        total = broadband.integrate_broadband_fluxes(spectral, g_weights, widths)
        np.testing.assert_allclose(total.upward, 5.670374419e-5 * 2500.0**4, rtol=1.5e-6)
        np.testing.assert_allclose(total.downward, 0.0, atol=1e-10)
        np.testing.assert_allclose(total.incident, jnp.sum(star * widths) * 0.4 / 3.5**2)
        np.testing.assert_allclose(total.net, total.upward - total.incident)
        assert broadband.bottom_leakage_fraction(total) == 1.0


def test_absorbed_beam_sign_energy_and_mu0_attenuation(broadband):
    boundaries = jnp.array([0.01, 0.1, 1.0])
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    dtau = jnp.array([[0.2, 0.8], [0.7, 1.3]])
    temperature = jnp.array([2000.0, 2400.0])
    incident = broadband.irradiation_flux(jnp.array([100.0, 300.0]), 2.0, 0.4)
    mus, angular_weights = _angles()
    fluxes = broadband.absorption_fluxes(
        dtau, temperature, 2700.0, pressure, boundaries,
        lambda values: jnp.zeros((len(values), 2)), mus, angular_weights,
        incident, mu0=0.25,
    )
    expected = incident * np.exp(-np.vstack((np.zeros(2), np.cumsum(dtau, axis=0))) / 0.25)
    np.testing.assert_allclose(fluxes.incident, expected)
    np.testing.assert_allclose(fluxes.net, -expected)
    heating = np.diff(fluxes.net, axis=0)
    assert np.all(heating > 0.0)
    np.testing.assert_allclose(heating.sum(axis=0), incident - expected[-1])
    integrated = broadband.Fluxes(*(jnp.sum(item, axis=-1) for item in fluxes))
    np.testing.assert_allclose(
        broadband.bottom_leakage_fraction(integrated), expected[-1].sum() / incident.sum(),
    )
    no_beam = broadband.Fluxes(*(jnp.zeros(3) for _ in range(4)))
    assert broadband.bottom_leakage_fraction(no_beam) == 0.0


def test_rorr_gray_bands_match_resolved_transfer_and_temperature_gradient(broadband):
    edges = _bands(300.0, 50000.0, count=12)
    nodes, band_weights = broadband.make_band_quadrature(edges)
    widths = jnp.asarray(np.diff(edges).ravel())
    boundaries = jnp.array([0.01, 0.08, 0.9, 8.0])
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = jnp.array([1900.0, 2200.0, 2700.0])
    mus, angular_weights = _angles()
    g_weights = jnp.array([0.2, 0.5, 0.3])
    # Constant k within each band provides an exact CKD/LBL comparison even
    # when the bands are broad and the source varies substantially within them.
    first = jnp.array([0.03, 0.2, 0.8])[:, None, None] * jnp.linspace(0.5, 2.0, 12)
    second = jnp.array([0.01, 0.1, 0.5])[:, None, None] * jnp.linspace(2.0, 0.5, 12)
    species = jnp.stack((jnp.broadcast_to(first, (3, 3, 12)),
                         jnp.broadcast_to(second, (3, 3, 12))))
    dtau = mix_ckd_rorr(species, g_weights)
    incident = broadband.irradiation_flux(
        broadband.band_planck_flux(6400.0, nodes, band_weights), 3.5, 0.5,
    )

    def ckd_result(values):
        spectral = broadband.ckd_absorption_fluxes(
            dtau, values, 3000.0, pressure, boundaries, nodes, band_weights,
            mus, angular_weights, incident, 0.6,
        )
        return broadband.integrate_broadband_fluxes(spectral, g_weights, widths)

    total = jax.jit(ckd_result)(temperature)
    resolved_nu = nodes.ravel()
    resolved_depth = jnp.repeat(dtau[:, 0, :], nodes.shape[1], axis=-1)
    resolved = broadband.absorption_fluxes(
        resolved_depth, temperature, 3000.0, pressure, boundaries,
        lambda values: piBarr(values, resolved_nu), mus, angular_weights,
        broadband.irradiation_flux(piBarr(jnp.array([6400.0]), resolved_nu)[0], 3.5, 0.5),
        mu0=0.6,
    )
    resolved_weights = (widths[:, None] * band_weights).ravel()
    for ckd, lbl in zip(total, resolved):
        np.testing.assert_allclose(ckd, jnp.sum(lbl * resolved_weights, axis=-1), rtol=2e-14, atol=1e-6)
    direction = jnp.array([1.0, -0.5, 0.8])
    _, derivative = jax.jvp(lambda values: ckd_result(values).net, (temperature,), (direction,))
    delta = 0.02
    fd = (ckd_result(temperature + delta * direction).net
          - ckd_result(temperature - delta * direction).net) / (2.0 * delta)
    np.testing.assert_allclose(derivative, fd, rtol=2e-8)
