"""Offline numerical checks for the Allard density expansion and hybrid core."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import voigt_profile

from exojax.database.alkali import AllardProfile
from exojax.database.core.broadening import doppler_sigma
from exojax.opacity import OpaAlkaliTable
from exojax.opacity.allard import _wing, density_expansion
from exojax.utils.constants import kB


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def profiles(monkeypatch):
    offsets = np.array([-100., -30., -20., 0., 20., 30., 100.])
    data = {}
    for line, wavelength, strength in [("D1", 5897.558, 2.82e-13),
                                       ("D2", 5891.582, 5.64e-13)]:
        data[line] = [AllardProfile(
            temperature=T, wavelength=wavelength, density=1e21,
            volume=2., normalization=strength,
            width=20. * (T / 1000.)**0.4, shift=0.5,
            offsets=offsets, coefficients=np.tile([0.002, -0.0001], (7, 1)),
            red_offsets=np.array([-1000., -500., -100.]) if line == "D1" else None,
            red_cross_sections=np.array([1e-19, 2e-18, 3e-17]) if line == "D1" else None,
        ) for T in [500., 1000., 2000.]]
    monkeypatch.setattr("exojax.database.alkali.load_allard2019", lambda path: data)
    return data


def test_density_powers_and_zero_density_derivative():
    coefficients = jnp.array([[2., -3., 5.], [4., 0., 0.]])
    q = 0.01
    expected = np.array([2*q - 3*q*q + 5*q**3, 4*q]) * np.exp(-2*q) * 3.
    np.testing.assert_allclose(density_expansion(coefficients, q, 2., 3.), expected)
    np.testing.assert_array_equal(density_expansion(coefficients, 0., 2., 3.), [0., 0.])
    derivative = jax.jacfwd(lambda q: density_expansion(coefficients, q, 2., 3.))(0.)
    np.testing.assert_allclose(derivative, [6., 12.])


def test_wing_support_signed_coefficients_and_far_red(profiles):
    p = profiles["D1"][1]
    offsets = jnp.array([-1001., -1000., -500., -100., 100., 101.])
    density = 1e19
    actual = _wing(p, offsets, density)
    tabulated = p.normalization * np.exp(-0.02) * (0.002*0.01 - 0.0001*0.01**2)
    np.testing.assert_allclose(actual, [0., 1e-21, 2e-20, tabulated, tabulated, 0.])
    # Clip the completed expansion, never individual signed coefficients.
    negative = replace(p, coefficients=-np.ones((7, 2)))
    np.testing.assert_array_equal(_wing(negative, jnp.array([-30., 30.]), density), [0., 0.])


def test_core_matches_scipy_with_table_shift_and_natural_width(profiles):
    center = 1e8 / profiles["D1"][1].wavelength
    grid = center + np.array([-0.02, 0., 0.02])
    opa = OpaAlkaliTable(grid, "unused", vmr_perturber=0.8)
    T, density = 1000., 1e19
    P = density*kB*T/1e6 / 0.8
    expected = np.zeros(3)
    for line in ("D1", "D2"):
        p = profiles[line][1]
        nu = 1e8 / p.wavelength
        natural = (2 if line == "D1" else 1)*p.normalization*nu**2
        expected += p.normalization * voigt_profile(
            grid - nu - p.shift*0.01, float(doppler_sigma(nu, T, opa.mass)),
            p.width*0.01 + natural,
        )
    np.testing.assert_allclose(jax.jit(opa.xsvector)(T, P), expected, rtol=2e-6)


def test_temperature_interpolation_jit_layers_and_gradients(profiles):
    opa = OpaAlkaliTable(np.linspace(16880., 17060., 51), "unused")
    T, P = 875., 1.2
    density = P*1e6/(kB*T)
    expected = np.zeros(51)
    for line in ("D1", "D2"):
        expected += (0.25 * np.asarray(opa._at_temperature(line, profiles[line][0], T, density))
                     + 0.75 * np.asarray(opa._at_temperature(line, profiles[line][1], T, density)))
    actual = jax.jit(opa.xsvector)(T, P)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    layers = jax.jit(opa.xsmatrix)(jnp.array([T, 1000.]), jnp.array([P, 0.]))
    np.testing.assert_allclose(layers[0], actual, rtol=2e-6)
    np.testing.assert_allclose(layers[1], opa.xsvector(1000., 0.), rtol=2e-6)
    signal = lambda t, p: jnp.sum(opa.xsvector(t, p))*1e14
    derivatives = jax.jit(jax.grad(signal, (0, 1)))(T, P)
    expected_derivatives = [(signal(T+0.01, P)-signal(T-0.01, P))/0.02,
                            (signal(T, P+1e-5)-signal(T, P-1e-5))/2e-5]
    np.testing.assert_allclose(derivatives, expected_derivatives, rtol=2e-5)
    # Direct reverse-mode derivatives must work without rescaling the output.
    direct = jax.jit(jax.jacrev(opa.xsvector, (0, 1)))(T, P)
    finite = [(opa.xsvector(T+0.01, P)-opa.xsvector(T-0.01, P))/0.02,
              (opa.xsvector(T, P+1e-5)-opa.xsvector(T, P-1e-5))/2e-5]
    for derivative, reference in zip(direct, finite):
        np.testing.assert_allclose(derivative, reference, rtol=2e-5, atol=1e-30)


@pytest.mark.parametrize("T,P", [(499., 1.), (2001., 1.), (1000., -1.),
                                 (1000., 1e10), (np.nan, 1.), (1000., np.inf)])
def test_invalid_domain_is_nan_under_jit(profiles, T, P):
    opa = OpaAlkaliTable(np.array([16960., 16970.]), "unused")
    assert np.all(np.isnan(jax.jit(opa.xsvector)(T, P)))


def test_zero_pressure_and_spectral_support(profiles):
    opa = OpaAlkaliTable(np.array([12000., 16956., 16973., 22000.]), "unused")
    actual = jax.jit(opa.xsvector)(1000., 0.)
    assert np.all(np.isfinite(actual))
    assert actual[1] > 0 and actual[2] > 0
    assert actual[0] == actual[-1] == 0


def test_join_value_and_first_derivative_are_continuous(profiles):
    p = profiles["D1"][1]
    center = 1e8/p.wavelength
    for offset in [-30., -20., 20., 30.]:
        epsilon = 1e-4
        opa = OpaAlkaliTable(center + offset + np.arange(-2, 3)*epsilon, "unused")
        y = np.asarray(opa._at_temperature("D1", p, 1000., 1e19))*1e16
        left, right = (y[2]-y[1])/epsilon, (y[3]-y[2])/epsilon
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=2e-3)


def test_constructor_and_layer_validation(profiles):
    for grid in [[], [2., 1.], [1., 1.], [np.nan], [-1.], [[1., 2.]]]:
        with pytest.raises(ValueError, match="nu_grid"):
            OpaAlkaliTable(grid, "unused")
    for fraction in [-0.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="vmr_perturber"):
            OpaAlkaliTable([16970.], "unused", vmr_perturber=fraction)
    for transition in [(0., 30.), (30., 20.), (10.,), (10., np.inf)]:
        with pytest.raises(ValueError, match="core_transition"):
            OpaAlkaliTable([16970.], "unused", core_transition=transition)
    with pytest.raises(ValueError, match="model"):
        OpaAlkaliTable([16970.], "unused", model="unknown")
    opa = OpaAlkaliTable([16970.], "unused")
    with pytest.raises(ValueError, match="scalar"):
        opa.xsvector(jnp.array([1000.]), 1.)
    with pytest.raises(ValueError, match="equal shape"):
        opa.xsmatrix(jnp.array([1000.]), jnp.array([1., 2.]))


def test_float64_required(profiles):
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="32bit mode is not allowed"):
        OpaAlkaliTable([16970.], "unused")
