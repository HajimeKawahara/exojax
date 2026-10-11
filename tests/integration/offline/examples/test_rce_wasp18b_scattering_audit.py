"""Verify stellar-scattering boundaries independently of the P7 large assets."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("rce_wasp18b_scattering_audit")
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("closure", ["quadrature", "hemispheric_mean"])
def test_zero_scattering_matches_shared_p7_boundaries(audit, closure):
    from _rce_broadband import absorption_fluxes

    absorption = jnp.array([[[0., .1, .8], [.01, 2., 7.]],
                            [[1e-12, .3, .2], [1., .4, 12.]]])
    horizontal = jnp.array([[4., 0., 9.]])
    actual = audit.stellar_scattering_fluxes(absorption, 0., horizontal, .37, closure=closure)
    # Non-isothermal sources demonstrate that the shared stellar convention
    # does not depend on P7's thermal angular quadrature or source reconstruction.
    reference = absorption_fluxes(absorption, jnp.array([2300., 2700.]), 3000.,
        jnp.array([.02, .2]), jnp.array([.01, .1, 1.]),
        lambda t: t[:, None, None]**4, jnp.array([.2, .8]), jnp.array([.5, .5]),
        incident_flux=horizontal, mu0=.37)
    np.testing.assert_array_equal(actual.upward, np.zeros((3, 2, 3)))
    np.testing.assert_array_equal(actual.downward, np.zeros((3, 2, 3)))
    np.testing.assert_allclose(actual.incident, reference.incident, rtol=2e-15, atol=0.)
    np.testing.assert_allclose(actual.net, -reference.incident, rtol=2e-15, atol=0.)


@pytest.mark.parametrize("closure", ["quadrature", "hemispheric_mean"])
@pytest.mark.parametrize("dtype,tiny,rtol", [(jnp.float64, 1e-290, 2e-13),
                                           (jnp.float32, 1e-30, 2e-6)])
def test_tiny_absorption_derivatives_match_beer_lambert(audit, closure, dtype, tiny, rtol):
    # Positive opacity floors outside scattering support must differentiate as
    # pure absorption, including when squaring the total depth would underflow.
    depth = jnp.array([[[0., tiny, 1e-20, .3]],
                       [[0., 2 * tiny, 2e-20, .6]]], dtype=dtype)
    horizontal, mu0 = jnp.asarray(7., dtype), .5

    def flux(scale):
        return audit.stellar_scattering_fluxes(depth * scale, jnp.zeros_like(depth),
                                               horizontal, mu0, closure=closure)

    def reference(scale):
        accumulated = jnp.concatenate((jnp.zeros_like(depth[:1]),
                                        jnp.cumsum(depth * scale, axis=0)))
        return -horizontal * jnp.exp(-accumulated / mu0)

    scale = jnp.asarray(1., dtype)
    value, tangent = jax.jvp(flux, (scale,), (scale,))
    expected, expected_tangent = jax.jvp(jax.jit(reference), (scale,), (scale,))
    np.testing.assert_allclose(value.net, expected, rtol=rtol, atol=0.)
    np.testing.assert_allclose(tangent.net, expected_tangent, rtol=rtol, atol=0.)
    np.testing.assert_array_equal(tangent.upward, 0.)
    np.testing.assert_array_equal(tangent.downward, 0.)
    # Compare every boundary, so ordinary-depth bands cannot hide tiny errors.
    np.testing.assert_allclose(jax.jacrev(lambda s: flux(s).net)(scale),
                               expected_tangent, rtol=rtol, atol=0.)


@pytest.mark.parametrize("closure,rate", [("quadrature", np.sqrt(3.)), ("hemispheric_mean", 2.)])
def test_zero_scattering_retains_first_scattering_derivative(audit, closure, rate):
    absorption = jnp.array([[[1e-290, 1e-200, 1e-10, .3, 3.]]])
    horizontal, mu0 = 7., .5

    def reflection(scattering):
        return audit.stellar_scattering_fluxes(absorption, scattering,
            horizontal, mu0, closure=closure).upward[0, 0]

    # Integrate a single scattering event with direct-beam attenuation above
    # the event and absorption of the outgoing diffuse ray on its way to TOA.
    x = (rate + 1. / mu0) * np.asarray(absorption[0, 0])
    expected = horizontal / (2. * mu0) * (-np.expm1(-x) / x)
    value, tangent = jax.jvp(reflection, (0.,), (1.,))
    np.testing.assert_array_equal(value, 0.)
    np.testing.assert_allclose(tangent, expected, rtol=3e-13, atol=0.)
    np.testing.assert_allclose(jax.jacrev(reflection)(0.), expected, rtol=3e-13, atol=0.)


@pytest.mark.parametrize("closure", ["quadrature", "hemispheric_mean"])
def test_conservative_scattering_preserves_energy_at_every_boundary(audit, closure):
    scattering = jnp.array([[[0., 1e-10, .01]], [[.01, .1, 1.]], [[.3, 1., 100.]]])
    horizontal, mu0 = jnp.array([[2., 3., 7.]]), .23
    flux = audit.stellar_scattering_fluxes(0., scattering, horizontal, mu0, closure=closure)
    np.testing.assert_array_equal(flux.downward[0], np.zeros((1, 3)))
    np.testing.assert_array_equal(flux.upward[-1], np.zeros((1, 3)))
    np.testing.assert_allclose(flux.net, np.broadcast_to(flux.net[:1], flux.net.shape), rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(flux.upward[0] + flux.downward[-1] + flux.incident[-1],
                               horizontal, rtol=2e-13, atol=0.)


@pytest.mark.parametrize("closure,rate", [("quadrature", np.sqrt(3.)), ("hemispheric_mean", 2.)])
def test_uniform_absorbing_slab_matches_independent_matrix_exponential(audit, closure, rate):
    absorption, scattering, mu0, horizontal = .3, .5, .63, 11.
    tau = absorption + scattering
    omega = scattering / tau
    gamma1, gamma2 = rate * (1. - omega / 2.), rate * omega / 2.
    # Independent ODE for [F_up, F_down, beam-normal direct irradiance].
    equation = np.array([[gamma1, -gamma2, -omega / 2.],
                         [gamma2, -gamma1, omega / 2.], [0., 0., -1. / mu0]])
    transfer = expm(equation * tau)
    normal = horizontal / mu0
    top_up = -transfer[0, 2] * normal / transfer[0, 0]
    bottom_state = transfer @ np.array([top_up, 0., normal])
    actual = audit.stellar_scattering_fluxes(jnp.array([[[absorption]]]),
        jnp.array([[[scattering]]]), horizontal, mu0, closure=closure)
    np.testing.assert_allclose(actual.upward[:, 0, 0], [top_up, 0.], rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(actual.downward[:, 0, 0], [0., bottom_state[1]], rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(actual.incident[:, 0, 0], [horizontal, mu0 * bottom_state[2]], rtol=2e-14)
    assert float(actual.net[-1, 0, 0] - actual.net[0, 0, 0]) > 0.


def test_subdivision_and_thin_first_scattering(audit):
    whole = audit.stellar_scattering_fluxes(jnp.array([[[.7, 0.]]]),
        jnp.array([[[.3, 1e-9]]]), jnp.array([[5., 7.]]), .4)
    split = audit.stellar_scattering_fluxes(jnp.full((50, 1, 1), .7 / 50),
        jnp.full((50, 1, 1), .3 / 50), 5., .4)
    for entire, divided in zip(whole, split):
        np.testing.assert_allclose(entire[[0, -1], 0, 0], divided[[0, -1], 0, 0],
                                   rtol=3e-14, atol=1e-14)
    # g=0 scatters half the first-event energy into the upper hemisphere.
    np.testing.assert_allclose(whole.upward[0, 0, 1], .5 * 7. * 1e-9 / .4, rtol=3e-9)


def test_report_integrates_probability_and_wavenumber_weights(audit):
    shape = (3, 2, 2)
    incident = np.broadcast_to(np.array([10., 6., 2.])[:, None, None], shape)
    upward = np.broadcast_to(np.array([2., 1., 0.])[:, None, None], shape)
    downward = np.broadcast_to(np.array([0., 1., 1.])[:, None, None], shape)
    flux = audit.Fluxes(upward, downward, incident, upward - downward - incident)
    report = audit.integrated_stellar_report(flux, jnp.array([.3, .7]), jnp.array([2., 3.]))
    np.testing.assert_allclose(report["layer_heating_cgs"], [10., 15.])
    assert report["reflected_fraction"] == pytest.approx(.2)
    assert report["bottom_reaching_fraction"] == pytest.approx(.3)
    assert report["atmosphere_absorbed_fraction"] == pytest.approx(.5)
    assert report["energy_closure_residual_over_incident"] == pytest.approx(0.)
    comparison = audit.compare_stellar_reports(report, report)
    assert comparison["max_absolute_layer_heating_difference_over_incident"] == 0.
    assert all(value == 0. for value in comparison["max_absolute_boundary_difference_over_incident"].values())
