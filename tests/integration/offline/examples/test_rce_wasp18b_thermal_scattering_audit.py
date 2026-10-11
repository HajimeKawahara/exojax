"""Check thermal boundaries and source refinement without large opacity assets."""

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
    yield importlib.import_module("rce_wasp18b_thermal_scattering_audit")
    jax.config.update("jax_enable_x64", previous)


def test_zero_scattering_matches_independent_absorption_recurrence(audit):
    depth = np.array([[[0., 1e-12, .1], [.4, 1., 100.]],
                      [[.03, 3., 10.], [2., 0., 1e-9]]])
    source = np.array([[[2., 3., 4.]], [[7., 5., 9.]]])
    bottom = np.array([[11., 8., 6.]])
    actual = audit.thermal_fluxes(depth, 0., source, bottom)
    expected = audit.constant_absorption_reference(depth, source, bottom)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=3e-14, atol=2e-15)
    np.testing.assert_array_equal(actual.downward[0], np.zeros((2, 3)))
    np.testing.assert_array_equal(actual.upward[-1], np.broadcast_to(bottom, (2, 3)))


def test_conservative_slab_preserves_net_flux_and_black_boundaries(audit):
    scatter = np.array([[[0., 1e-12, .1]], [[.3, 1., 100.]], [[.7, 2., .2]]])
    bottom = np.array([[2., 4., 7.]])
    flux = audit.thermal_fluxes(0., scatter, 1e6, bottom)
    expected = bottom / (1. + np.sum(scatter, axis=0))
    np.testing.assert_allclose(flux.net, np.broadcast_to(expected, flux.net.shape), rtol=3e-13, atol=2e-14)
    np.testing.assert_array_equal(flux.downward[0], np.zeros((1, 3)))
    np.testing.assert_array_equal(flux.upward[-1], bottom)
    np.testing.assert_allclose(flux.upward[0] + flux.downward[-1], bottom, rtol=3e-14)


def test_isothermal_cavity_is_in_equilibrium(audit):
    absorption = np.array([[[.0, .1, 10.]], [[2., 0., 1e-12]]])
    scatter = np.array([[[.3, .0, .1]], [[.2, 4., 100.]]])
    source = np.array([[3., 5., 7.]])
    flux = audit.thermal_fluxes(absorption, scatter, source, source, incoming_top=source)
    for value in (flux.upward, flux.downward):
        np.testing.assert_allclose(value, np.broadcast_to(source, value.shape), rtol=5e-14, atol=1e-14)
    np.testing.assert_allclose(flux.net, 0., rtol=0., atol=1e-13)


def linear_reference(absorption, scattering, boundary, center, fraction):
    """Solve the linear-source two-stream ODE using an independent exponential."""
    segments = []
    for a, s, upper, lower, middle, f in zip(absorption, scattering, boundary[:-1], boundary[1:], center, fraction):
        total, omega = a + s, s / (a + s)
        for depth, left, right in ((total * f, upper, middle), (total * (1. - f), middle, lower)):
            gamma1, gamma2 = 2. - omega, omega
            # State is [F_up, F_down, pi*B, 1], with optical depth downward.
            equation = np.array([[gamma1, -gamma2, -2. * (1. - omega), 0.],
                                 [gamma2, -gamma1, 2. * (1. - omega), 0.],
                                 [0., 0., 0., (right - left) / depth],
                                 [0., 0., 0., 0.]])
            transfer = expm(equation * depth)
            segments.append((transfer[:2, :2], transfer[:2, 2:] @ [left, 1.]))
    matrix, offset = np.eye(2), np.zeros(2)
    for own_matrix, own_offset in segments:
        matrix, offset = own_matrix @ matrix, own_matrix @ offset + own_offset
    flux = np.array([(boundary[-1] - offset[0]) / matrix[0, 0], 0.])
    interfaces = [flux]
    for index, (matrix, offset) in enumerate(segments):
        flux = matrix @ flux + offset
        if index % 2:
            interfaces.append(flux)
    return np.array(interfaces).T


@pytest.mark.parametrize("scatter", [np.array([0., 0.]), np.array([.2, .3])])
def test_midpoint_sources_converge_to_independent_linear_solution(audit, scatter):
    absorption = np.array([.15, .4])
    boundary, center, fraction = np.array([2., 4., 7.]), np.array([3., 6.]), np.array([.3, .6])
    expected = linear_reference(absorption, scatter, boundary, center, fraction)
    errors = []
    for n in (1, 2, 4):
        actual, _ = audit.paired_fluxes(absorption[:, None, None], scatter[:, None, None],
            boundary[:, None, None], center[:, None, None], fraction, subdivisions=n)
        errors.append(np.max(np.abs(np.asarray(actual[:2])[:, :, 0, 0] - expected)))
    assert errors[0] > 1e-3
    assert .20 < errors[1] / errors[0] < .30
    assert .20 < errors[2] / errors[1] < .30
    if not np.any(scatter):
        up, down = audit.rtrun_emis_pureabs_ibased_linsap_fluxes(
            absorption[:, None, None], boundary[:, None, None], jnp.array([.5]), jnp.array([1.]),
            source_center=center[:, None, None], upper_fraction=fraction)
        np.testing.assert_allclose(np.asarray([up, down])[:, :, 0, 0], expected, rtol=2e-14, atol=2e-14)


def test_constant_sources_are_invariant_under_subdivision(audit):
    a, s = np.array([.1, 1.])[:, None, None], np.array([.2, .4])[:, None, None]
    boundary, center, fraction = np.full((3, 1, 1), 5.), np.full((2, 1, 1), 5.), np.array([.2, .7])
    expected = (audit.thermal_fluxes(a, s, center, boundary[-1]), audit.thermal_fluxes(a, 0., center, boundary[-1]))
    for n in (1, 2, 4):
        actual = audit.paired_fluxes(a, s, boundary, center, fraction, subdivisions=n)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_band_blocking_preserves_integrals_and_separates_approximations(audit):
    from exojax.rt.rtransfer import initialize_gaussian_quadrature

    absorption = np.array([[[.1, .3, 2.], [.2, .4, 3.]], [[.3, .2, .1], [.4, .3, .2]]])
    boundary = np.array([[[2., 4., 6.]], [[3., 5., 7.]], [[5., 7., 9.]]])
    center = (boundary[:-1] + boundary[1:]) / 2.
    mus, weights = initialize_gaussian_quadrature(8)
    arguments = (absorption, np.zeros((2, 1, 3)), boundary, center, np.array([.3, .7]),
                 np.array([.4, .6]), np.array([1., 2., 3.]), mus, weights, 100.)
    full = audit.audit_arrays(*arguments, band_batch=3)
    blocked = audit.audit_arrays(*arguments, band_batch=1)
    for n in (1, 2, 4):
        for key in ("upward", "downward", "net"):
            np.testing.assert_allclose(full["cases"][str(n)]["scattering_on"]["boundary_flux_cgs"][key],
                                       blocked["cases"][str(n)]["scattering_on"]["boundary_flux_cgs"][key], rtol=3e-14)
            np.testing.assert_array_equal(full["cases"][str(n)]["on_minus_off"]["boundary_flux_difference_cgs"][key], 0.)
    source_errors = [full["cases"][str(n)]["off_minus_linear_hm"]["max_boundary_difference_over_incident"]["net"] for n in (1, 2, 4)]
    assert source_errors[2] < source_errors[1] < source_errors[0]
    assert full["closure_difference_hm_minus_p7_angular"]["max_boundary_difference_over_incident"]["net"] > 0.
    assert max(full["limiting_checks"].values()) < 5e-12
