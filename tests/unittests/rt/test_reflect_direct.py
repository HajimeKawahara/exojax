"""Physical limits and differentiation of reflection from a direct beam."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from exojax.rt import ArtReflectPure
from exojax.rt.direct_sfm import _direct_layer_sources
from exojax.rt.twostream import (
    set_scat_trans_absorption_coeffs,
    solve_fluxadding_twostream_fluxes,
)

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


@pytest.fixture(autouse=True)
def _double_precision():
    with enable_x64():
        yield


def _art(nlayer=2, nnus=3):
    return ArtReflectPure(nlayer=nlayer, nu_grid=jnp.arange(nnus) + 1000.0)


def test_direct_absorbing_black_atmosphere_returns_zero():
    dtau = jnp.array([[0.0, 0.3, 40.0], [0.0, 0.7, 60.0]])
    actual = _art().run_direct(
        dtau, jnp.zeros_like(dtau), jnp.zeros(3), jnp.ones(3), 0.4, 0.7
    )

    np.testing.assert_array_equal(actual, np.zeros(3))


@pytest.mark.parametrize("transparent", [True, False])
def test_direct_lambert_surface_normalization_and_absorption(transparent):
    dtau = jnp.array([[0.0, 0.2, 0.7], [0.0, 0.3, 1.1]])
    if transparent:
        dtau = jnp.zeros_like(dtau)
    albedo = jnp.array([0.2, 0.4, 0.8])
    incident = jnp.array([1.0, 2.0, 3.0])
    mu_in, mu_out = 0.3, 0.7

    actual = _art().run_direct(
        dtau, jnp.zeros_like(dtau), albedo, incident, mu_in, mu_out
    )
    expected = (
        albedo
        * mu_in
        * incident
        / np.pi
        * jnp.exp(-jnp.sum(dtau, axis=0) * (1.0 / mu_in + 1.0 / mu_out))
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)


@pytest.mark.parametrize("nlayer", [2, 7])
@pytest.mark.parametrize("phase_function", ["isotropic", "rayleigh"])
@pytest.mark.parametrize("mu_in,mu_out,azimuth", [(1.0, 1.0, 0.0), (0.2, 0.7, 1.1)])
def test_direct_weak_scattering_matches_analytic_single_scattering(
    nlayer, phase_function, mu_in, mu_out, azimuth
):
    total_tau = jnp.array([0.0, 1.0e-8, 1.0, 100.0])
    fractions = jnp.arange(1.0, nlayer + 1.0)
    fractions = fractions / jnp.sum(fractions)
    dtau = fractions[:, None] * total_tau[None, :]
    omega = 1.0e-8
    incident = jnp.array([1.0, 2.0, 3.0, 4.0])

    actual = _art(nlayer, 4).run_direct(
        dtau,
        jnp.full_like(dtau, omega),
        jnp.zeros(4),
        incident,
        mu_in,
        mu_out,
        relative_azimuth=azimuth,
        phase_function=phase_function,
    )
    cos_phase = mu_in * mu_out + np.sqrt(1.0 - mu_in**2) * np.sqrt(
        1.0 - mu_out**2
    ) * np.cos(azimuth)
    phase = 1.0 if phase_function == "isotropic" else 0.75 * (1.0 + cos_phase**2)
    expected = (
        omega
        * incident
        / (4.0 * np.pi)
        * phase
        * mu_in
        / (mu_in + mu_out)
        * -jnp.expm1(-total_tau * (1.0 / mu_in + 1.0 / mu_out))
    )

    # Multiple scattering contributes only at second order in omega.
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=0.0)


def test_direct_rayleigh_backscatter_and_azimuth_dependence():
    art = _art()
    dtau = jnp.full((2, 3), 0.1)
    inputs = (dtau, jnp.full_like(dtau, 1.0e-8), jnp.zeros(3), jnp.ones(3))
    mu = 1.0 / np.sqrt(2.0)

    isotropic = art.run_direct(*inputs, mu, mu, phase_function="isotropic")
    backscatter = art.run_direct(*inputs, mu, mu, relative_azimuth=0.0)
    right_angle = art.run_direct(*inputs, mu, mu, relative_azimuth=np.pi)

    np.testing.assert_allclose(backscatter / isotropic, 1.5, rtol=2.0e-7)
    np.testing.assert_allclose(right_angle / isotropic, 0.75, rtol=2.0e-7)


def test_direct_jit_and_gradients_at_transparency_resonance_and_conservative_limit():
    art = _art(nnus=4)
    dtau = jnp.array([[0.0, 0.3, 0.7, 20.0], [0.0, 0.5, 1.1, 80.0]])
    omega = jnp.broadcast_to(jnp.array([0.0, 2.0 / 3.0, 0.8, 1.0]), dtau.shape)

    def reflected_sum(optical_depth, scattering_albedo):
        return jnp.sum(
            art.run_direct(
                optical_depth,
                scattering_albedo,
                jnp.full(4, 0.2),
                jnp.ones(4),
                1.0,
                0.6,
            )
        )

    value, gradients = jax.jit(jax.value_and_grad(reflected_sum, argnums=(0, 1)))(
        dtau, omega
    )
    assert np.isfinite(value)
    for derivative in gradients:
        assert np.all(np.isfinite(derivative))

    # Check an interior derivative of each independent optical property.
    step = 1.0e-5
    perturbation = jnp.zeros_like(dtau).at[0, 2].set(step)
    finite_tau = (
        reflected_sum(dtau + perturbation, omega)
        - reflected_sum(dtau - perturbation, omega)
    ) / (2.0 * step)
    finite_omega = (
        reflected_sum(dtau, omega + perturbation)
        - reflected_sum(dtau, omega - perturbation)
    ) / (2.0 * step)
    np.testing.assert_allclose(gradients[0][0, 2], finite_tau, rtol=2.0e-5, atol=1.0e-9)
    np.testing.assert_allclose(
        gradients[1][0, 2], finite_omega, rtol=2.0e-5, atol=1.0e-9
    )


def test_direct_full_phase_lambert_sphere_has_two_thirds_surface_albedo():
    art = _art(nnus=2)
    dtau = jnp.zeros((2, 2))
    albedo = jnp.array([0.3, 0.8])
    incident = jnp.array([2.0, 3.0])
    abscissa, weights = np.polynomial.legendre.leggauss(8)
    mus = jnp.asarray((abscissa + 1.0) / 2.0)
    weights = jnp.asarray(weights / 2.0)

    intensities = jax.jit(
        jax.vmap(lambda mu: art.run_direct(dtau, dtau, albedo, incident, mu, mu))
    )(mus)
    geometric_albedo = (
        2.0
        * np.pi
        * jnp.sum(weights[:, None] * mus[:, None] * intensities, axis=0)
        / incident
    )

    np.testing.assert_allclose(geometric_albedo, 2.0 * albedo / 3.0, rtol=1.0e-12)


@pytest.mark.parametrize(
    "mu_in,mu_out", [(0.0, 0.5), (0.5, -0.1), (1.1, 0.5), (0.5, np.nan)]
)
def test_direct_rejects_invalid_angles(mu_in, mu_out):
    dtau = jnp.ones((2, 3))
    with pytest.raises(ValueError):
        _art().run_direct(dtau, dtau, jnp.zeros(3), jnp.ones(3), mu_in, mu_out)


def test_direct_rejects_unknown_phase_function():
    dtau = jnp.ones((2, 3))
    with pytest.raises(ValueError, match="phase"):
        _art().run_direct(
            dtau,
            dtau,
            jnp.zeros(3),
            jnp.ones(3),
            0.5,
            0.5,
            phase_function="unknown",
        )


@pytest.mark.parametrize("omega", [0.5, 2.0 / 3.0, 1.0])
@pytest.mark.parametrize("mu_in", [0.2, 1.0])
def test_direct_layer_sources_match_matrix_exponential(omega, mu_in):
    dtau = jnp.array([0.0, 0.1, 3.0])
    gamma1 = np.sqrt(3.0) * (1.0 - omega / 2.0)
    gamma2 = np.sqrt(3.0) * omega / 2.0
    gamma3 = 0.5
    actual = _direct_layer_sources(dtau, gamma1, gamma2, omega, mu_in, gamma3)

    # Propagate [F_plus, F_minus, beam] through a homogeneous layer, then
    # impose zero incoming diffuse flux at both boundaries.
    equation = np.array(
        [
            [gamma1, -gamma2, -omega * gamma3],
            [gamma2, -gamma1, omega * (1.0 - gamma3)],
            [0.0, 0.0, -1.0 / mu_in],
        ]
    )
    expected = []
    for optical_depth in np.asarray(dtau):
        transfer = expm(equation * optical_depth)
        source_plus = -transfer[0, 2] / transfer[0, 0]
        source_minus = transfer[1, 0] * source_plus + transfer[1, 2]
        expected.append((source_plus, source_minus))

    np.testing.assert_allclose(
        actual, np.asarray(expected).T, rtol=1.0e-11, atol=1.0e-14
    )


@pytest.mark.parametrize("mu_in", [0.2, 1.0])
def test_direct_conservative_layer_stack_conserves_beam_energy(mu_in):
    dtau = jnp.array([[0.0, 1.0e-8, 0.01], [0.1, 0.3, 0.5], [0.9, 2.7, 10.0]])
    omega = jnp.ones_like(dtau)
    gamma = jnp.full_like(dtau, np.sqrt(3.0) / 2.0)
    trans, scat, absorption = set_scat_trans_absorption_coeffs(gamma, gamma, dtau)
    source_plus, source_minus = _direct_layer_sources(
        dtau, gamma, gamma, omega, mu_in, 0.5
    )
    tau = jnp.concatenate((jnp.zeros_like(dtau[:1]), jnp.cumsum(dtau, axis=0)))
    beam = jnp.exp(-tau / mu_in)

    flux_plus, flux_minus = solve_fluxadding_twostream_fluxes(
        trans,
        scat,
        jnp.zeros_like(dtau),
        jnp.zeros(3),
        jnp.zeros(3),
        absorption_coeff=absorption,
        source_plus=source_plus * beam[:-1],
        source_minus=source_minus * beam[:-1],
    )

    # All incident energy leaves the top or reaches the black lower boundary.
    np.testing.assert_allclose(
        flux_plus[0] + flux_minus[-1] + mu_in * beam[-1], mu_in, rtol=1.0e-12
    )
