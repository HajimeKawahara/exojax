import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad, jacfwd, jit

from exojax.rt.flux import (
    direct_beam_fluxes,
    integrate_ckd_flux,
    reconstruct_boundary_temperature,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.rtransfer import (
    initialize_gaussian_quadrature,
    rtrun_emis_pureabs_ibased_linsap,
)


def test_boundary_reconstruction_including_top_and_bottom():
    pressure = jnp.array([1.0, 4.0, 16.0])
    boundaries = jnp.array([0.5, 2.0, 8.0, 32.0])
    temperature = 300.0 * pressure**0.2
    bottom = 300.0 * boundaries[-1]**0.2
    actual = jit(reconstruct_boundary_temperature)(pressure, boundaries, temperature, bottom)
    expected = np.append(temperature[0], 300.0 * np.asarray(boundaries[1:])**0.2)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("spectral_shape", [(), (3,), (2, 3)])
def test_toa_matches_existing_linsap(spectral_shape):
    mus, weights = initialize_gaussian_quadrature(8)
    size = int(np.prod(spectral_shape))
    dtau = jnp.linspace(0.0, 4.0, 4 * size).reshape((4,) + spectral_shape)
    source = jnp.linspace(1.0, 5.0, 5 * size).reshape((5,) + spectral_shape)

    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, source, mus, weights
    )
    expected = rtrun_emis_pureabs_ibased_linsap(
        dtau.reshape(4, size), source.reshape(5, size), mus, weights
    )

    assert upward.shape == downward.shape == source.shape
    np.testing.assert_allclose(upward[0], expected.reshape(spectral_shape), rtol=2e-6)
    np.testing.assert_array_equal(downward[0], jnp.zeros(spectral_shape))


def test_transparent_atmosphere_preserves_both_boundary_fluxes():
    mus, weights = initialize_gaussian_quadrature(8)
    dtau = jnp.zeros((4, 2, 3))
    source = jnp.full((5, 1, 3), 100.0)
    top = jnp.array([1.0, 2.0, 3.0])
    bottom = jnp.array([[4.0], [5.0]])

    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, source, mus, weights, incoming_top=top, outgoing_bottom=bottom
    )

    np.testing.assert_allclose(
        upward, jnp.broadcast_to(bottom, upward.shape), rtol=1e-6
    )
    np.testing.assert_allclose(
        downward, jnp.broadcast_to(top, downward.shape), rtol=1e-6
    )


def test_isothermal_atmosphere_matches_analytic_solution_at_all_interfaces():
    mus, weights = initialize_gaussian_quadrature(8)
    dtau = jnp.array([0.0, 1.0e-8, 0.2, 1.0, 10.0], dtype=jnp.float32)
    source = jnp.float32(3.0)
    top, bottom = 0.5, 2.0
    tau = np.concatenate(([0.0], np.cumsum(np.asarray(dtau, dtype=np.float64))))
    transmission_up = np.exp(-(tau[-1] - tau[:, None]) / mus)
    transmission_down = np.exp(-tau[:, None] / mus)
    expected_up = np.sum(
        2 * mus * weights * (3 + (bottom - 3) * transmission_up), axis=1
    )
    expected_down = np.sum(
        2 * mus * weights * (3 + (top - 3) * transmission_down), axis=1
    )

    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, source, mus, weights, incoming_top=top, outgoing_bottom=bottom
    )

    np.testing.assert_allclose(upward, expected_up, rtol=2e-6)
    np.testing.assert_allclose(downward, expected_down, rtol=2e-6)


def test_optically_thick_linear_source_retains_diffusion_flux():
    mus, weights = initialize_gaussian_quadrature(8)
    dtau = jnp.full(6, 10.0)
    tau = jnp.arange(7) * 10.0
    gradient = 1.0
    source = 1.0 + gradient * tau

    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, source, mus, weights
    )

    # Away from both boundaries, F_net = 4/3 * d(pi B)/d(tau).
    np.testing.assert_allclose(
        upward[2:-2] - downward[2:-2], 4 * gradient / 3, rtol=2e-5
    )


@pytest.mark.parametrize("depth", [0.0, 1.0e-8, 0.1, 10.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_slab_opacity_derivative_is_finite_and_matches_analytic_solution(depth, dtype):
    mus, weights = initialize_gaussian_quadrature(8)
    mus, weights = jnp.asarray(mus, dtype=dtype), jnp.asarray(weights, dtype=dtype)

    def outgoing_fluxes(optical_depth):
        upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
            optical_depth[None],
            jnp.array([3.0, 3.0], dtype=dtype),
            mus,
            weights,
            incoming_top=1.0,
            outgoing_bottom=2.0,
        )
        return jnp.array([upward[0], downward[-1]])

    derivative = jit(jacfwd(outgoing_fluxes))(dtype(depth))
    expected = 2 * np.sum(weights * np.exp(-depth / mus)) * np.array([1.0, 2.0])

    np.testing.assert_allclose(derivative, expected, rtol=3e-6, atol=1e-7)


def test_zero_opacity_nonuniform_source_derivative():
    mus, weights = initialize_gaussian_quadrature(8)
    mus = jnp.asarray(mus, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)

    def outgoing_fluxes(depth):
        upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
            depth[None],
            jnp.array([2.0, 4.0], dtype=jnp.float32),
            mus,
            weights,
            incoming_top=1.0,
        )
        return jnp.array([upward[0], downward[-1]])

    np.testing.assert_allclose(
        jacfwd(outgoing_fluxes)(jnp.float32(0)), [-2.0, 4.0], rtol=1e-6
    )


@pytest.mark.parametrize("fraction", [0.5, np.array([0.2, 0.7])])
def test_center_sources_match_explicitly_split_ckd_layers(fraction):
    mus, weights = initialize_gaussian_quadrature(8)
    dtau = jnp.linspace(0.0, 3.0, 12).reshape(2, 2, 3)
    boundary = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    center = jnp.array([[2.0, 3.0, 5.0], [5.0, 8.0, 10.0]])
    fractions = np.broadcast_to(fraction, (2,))
    split_dtau = jnp.stack(
        [
            fractions[0] * dtau[0],
            (1 - fractions[0]) * dtau[0],
            fractions[1] * dtau[1],
            (1 - fractions[1]) * dtau[1],
        ]
    )
    split_source = jnp.stack(
        [boundary[0], center[0], boundary[1], center[1], boundary[2]]
    )[:, None, :]

    actual = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau,
        boundary[:, None, :],
        mus,
        weights,
        incoming_top=0.5,
        outgoing_bottom=4.0,
        source_center=center[:, None, :],
        upper_fraction=fraction,
    )
    expected = rtrun_emis_pureabs_ibased_linsap_fluxes(
        split_dtau,
        split_source,
        mus,
        weights,
        incoming_top=0.5,
        outgoing_bottom=4.0,
    )

    for flux, split_flux in zip(actual, expected):
        assert flux.shape == (3, 2, 3)
        np.testing.assert_allclose(flux, split_flux[::2], rtol=1e-12)


def test_center_source_opacity_and_position_gradients():
    mus, weights = initialize_gaussian_quadrature(8)

    def flux(parameters):
        depth, center, fraction = parameters
        return rtrun_emis_pureabs_ibased_linsap_fluxes(
            depth[None],
            jnp.zeros(2),
            mus,
            weights,
            source_center=center,
            upper_fraction=fraction,
        )[0][0]

    parameters = jnp.array([0.7, 2.0, 0.3])
    step = 1.0e-5
    perturbations = step * np.eye(3)
    finite_difference = np.array(
        [
            (flux(parameters + delta) - flux(parameters - delta)) / (2 * step)
            for delta in perturbations
        ]
    )
    np.testing.assert_allclose(grad(flux)(parameters), finite_difference, rtol=1e-7)
    # A triangular source has integrated source * depth / 2 at vanishing depth.
    np.testing.assert_allclose(
        grad(flux)(jnp.array([0.0, 2.0, 0.3])), [2.0, 0.0, 0.0], atol=1e-12
    )


def test_direct_beam_uses_horizontal_incident_flux_and_reaches_surface():
    dtau = jnp.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    incident = jnp.array([2.0, 3.0])
    mu0 = 0.4

    flux = direct_beam_fluxes(dtau, incident, mu0)
    tau = np.concatenate((np.zeros((1, 2)), np.cumsum(dtau, axis=0)), axis=0)

    np.testing.assert_allclose(flux, incident * np.exp(-tau / mu0), rtol=2e-6)
    np.testing.assert_array_equal(flux[0], incident)
    assert np.all(flux[-1] > 0)
    derivative = grad(lambda depth: direct_beam_fluxes(depth[None], 2.0, mu0)[-1])
    assert derivative(jnp.float32(0)) == pytest.approx(-2.0 / mu0)


def test_ckd_integration_preserves_leading_dimensions_and_band_units():
    spectral_flux = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3)
    weights = jnp.array([0.25, 0.75])
    band_widths = jnp.array([2.0, 3.0, 5.0])

    actual = integrate_ckd_flux(spectral_flux, weights, band_widths)
    expected = np.sum(
        (
            0.25 * np.asarray(spectral_flux[:, 0])
            + 0.75 * np.asarray(spectral_flux[:, 1])
        )
        * np.asarray(band_widths),
        axis=1,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert integrate_ckd_flux(
        jnp.ones((2, 3)), weights, band_widths
    ) == pytest.approx(10.0)
