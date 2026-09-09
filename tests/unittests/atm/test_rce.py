"""Energy balance, dry complementarity, and explicit solver failures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce import reconstruct_boundary_temperature, rce_residual, solve_rce
from exojax.rt.flux import direct_beam_fluxes, rtrun_emis_pureabs_ibased_linsap_fluxes
from exojax.rt.rtransfer import initialize_gaussian_quadrature


def _toy_flux(temperature, bottom_temperature):
    log_t = jnp.log(jnp.append(temperature, bottom_temperature))
    return jnp.concatenate(((temperature[:1] / 300.0) ** 4, jnp.array([4.0, 1.0]) * jnp.diff(log_t)))


def _toy_solve(**kwargs):
    inputs = dict(
        pressure_bar=np.array([1.0, 4.0]),
        pressure_boundaries_bar=np.array([0.5, 2.0, 16.0]),
        temperature_initial=np.array([250.0, 350.0]),
        bottom_temperature_initial=400.0,
        internal_flux=1.0,
        radiative_flux=_toy_flux,
        adiabatic_gradient=0.25,
        flux_atol=1.0e-9,
        flux_rtol=1.0e-9,
        gradient_atol=1.0e-9,
    )
    inputs.update(kwargs)
    return solve_rce(**inputs)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("initial_mask", [[False, False], [True, True]])
def test_active_set_grows_and_shrinks_to_exact_solution(warm, initial_mask):
    factor = 2.0 if warm else 0.6
    result = _toy_solve(
        temperature_initial=factor * np.array([300.0, 350.0]),
        bottom_temperature_initial=factor * 450.0,
        convective_mask_initial=np.array(initial_mask),
    )
    assert result.converged, result.status
    assert result.domain_valid
    np.testing.assert_array_equal(result.convective_mask, [False, True])
    np.testing.assert_allclose(result.temperature, [300.0, 300.0 * np.exp(0.25)], rtol=1e-9)
    np.testing.assert_allclose(result.bottom_temperature, 300.0 * np.exp(0.25) * 4.0**0.25, rtol=1e-9)
    np.testing.assert_allclose(result.convective_flux, [0.0, 0.0, 1.0 - np.log(4.0) / 4.0], atol=2e-9)
    assert np.max(np.abs(result.flux_residual)) <= 2e-9
    assert np.max(result.gradient_residual) <= 1e-9
    assert np.max(np.abs(result.scaled_residual)) <= 1.0


def test_stable_column_and_fixed_mask_jacobian():
    result = _toy_solve(adiabatic_gradient=np.array([0.3, 1.0]))
    assert result.converged, result.status
    assert not np.any(result.convective_mask)
    np.testing.assert_allclose(result.radiative_flux, 1.0, atol=2e-9)
    np.testing.assert_array_equal(result.convective_flux, 0.0)
    log_t = jnp.log(jnp.array([300.0, 400.0, 500.0]))

    def residual(x):
        return rce_residual(x, jnp.array([1.0, 4.0]), 16.0, 1.0,
                            _toy_flux, 0.25, jnp.array([False, True]))

    direction = jnp.array([0.2, -0.1, 0.3])
    step = 1.0e-5
    finite_difference = (residual(log_t + step * direction) - residual(log_t - step * direction)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(residual)(log_t) @ direction, finite_difference, rtol=1e-8)


def test_boundary_reconstruction_including_top_and_bottom():
    pressure = jnp.array([1.0, 4.0, 16.0])
    boundaries = jnp.array([0.5, 2.0, 8.0, 32.0])
    temperature = 300.0 * pressure**0.2
    bottom = 300.0 * boundaries[-1]**0.2
    actual = jax.jit(reconstruct_boundary_temperature)(pressure, boundaries, temperature, bottom)
    expected = np.append(temperature[0], 300.0 * np.asarray(boundaries[1:])**0.2)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def _independent_gray_flux(source, dtau):
    """Direct NumPy formal solution for a single ray at mu=1/2."""
    depth = dtau / 0.5
    transmission = np.exp(-depth)
    gamma = (1.0 - transmission) / depth - transmission
    beta = 1.0 - (1.0 - transmission) / depth
    up, down = np.zeros_like(source), np.zeros_like(source)
    up[-1] = source[-1]
    for i in range(len(dtau) - 1, -1, -1):
        up[i] = transmission[i] * up[i+1] + beta[i] * source[i] + gamma[i] * source[i+1]
    for i in range(len(dtau)):
        down[i+1] = transmission[i] * down[i] + gamma[i] * source[i] + beta[i] * source[i+1]
    return up - down


@pytest.mark.parametrize(
    "stellar_flux,internal_flux", [(0.0, 1.0e4), (2.0e4, 1.0e4), (2.0e4, 0.0)]
)
def test_gray_radiative_equilibrium_against_independent_linear_solve(
    stellar_flux, internal_flux
):
    sigma = 5.670374419e-5
    boundaries = jnp.geomspace(0.01, 10.0, 5)
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    dtau = jnp.array([0.2, 0.5, 1.0, 2.0])
    stellar = direct_beam_fluxes(0.3 * dtau, stellar_flux, 0.6)

    def flux(temperature, bottom_temperature):
        tb = reconstruct_boundary_temperature(pressure, boundaries, temperature, bottom_temperature)
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(dtau, sigma * tb**4, jnp.array([0.5]), jnp.array([1.0]))
        return up - down - stellar

    matrix = np.column_stack([_independent_gray_flux(basis, np.asarray(dtau)) for basis in np.eye(5)])
    expected_source = np.linalg.solve(matrix, internal_flux + np.asarray(stellar))
    result = solve_rce(pressure, boundaries, jnp.full(4, 160.0), 220.0,
                       internal_flux, flux, adiabatic_gradient=10.0,
                       flux_atol=1e-6, flux_rtol=1e-10)
    assert result.converged, result.status
    actual_tb = reconstruct_boundary_temperature(pressure, boundaries, result.temperature, result.bottom_temperature)
    np.testing.assert_allclose(sigma * actual_tb**4, expected_source, rtol=1e-9)
    np.testing.assert_allclose(result.radiative_flux, internal_flux, atol=2e-6)
    assert not np.any(result.convective_mask)
    # Separate top and black-boundary energy balances, including transmitted starlight.
    np.testing.assert_allclose(_independent_gray_flux(sigma * np.asarray(actual_tb)**4, np.asarray(dtau))[[0, -1]],
                               internal_flux + np.asarray(stellar)[[0, -1]], rtol=1e-9)


def test_domain_guard_rejects_steps_before_radiation():
    checked_temperatures = []

    def valid(temperature, bottom):
        checked_temperatures.append(np.max(temperature))
        return np.all(temperature <= 200.0)

    result = _toy_solve(temperature_initial=np.array([200.0, 200.0]),
                        bottom_temperature_initial=200.0, valid_state=valid)
    assert not result.converged
    assert result.status == "domain_step_failed"
    assert result.domain_valid
    assert any(value > 200.0 for value in checked_temperatures)
    np.testing.assert_allclose(result.temperature, 200.0)
    with pytest.raises(ValueError, match="valid_state domain"):
        _toy_solve(valid_state=lambda t, tb: False)


def test_backtracking_stays_inside_valid_domain():
    result = _toy_solve(
        temperature_initial=np.array([180.0, 200.0]),
        bottom_temperature_initial=230.0,
        valid_state=lambda t, tb: np.all(t <= 450.0) and tb <= 650.0,
        convective_mask_initial=np.array([False, True]),
    )
    assert result.converged, result.status
    assert result.domain_valid


@pytest.mark.parametrize("kind,status", [
    ("iterations", "max_iterations"),
    ("masks", "max_active_set_iterations"),
    ("singular", "singular_jacobian"),
    ("nonfinite", "nonfinite_residual"),
])
def test_nonconvergence_is_explicit(kind, status):
    options = {
        "iterations": dict(max_iterations=1),
        "masks": dict(max_active_set_iterations=1),
        "singular": dict(radiative_flux=lambda t, tb: jnp.zeros(3)),
        "nonfinite": dict(radiative_flux=lambda t, tb: jnp.full(3, jnp.nan)),
    }
    result = _toy_solve(**options[kind])
    assert not result.converged
    assert result.status == status


def test_active_set_cycle_is_reported():
    def flux(temperature, bottom):
        gradient = jnp.log(bottom / temperature[0])
        return jnp.array([temperature[0], 1.5 - gradient])

    result = solve_rce(np.array([1.0]), np.array([0.5, np.e]), np.array([1.0]),
                       np.exp(0.4), 1.0, flux, adiabatic_gradient=0.25)
    assert not result.converged
    assert result.status == "active_set_cycle"


def test_nonfinite_flux_cannot_hide_in_an_active_connection():
    def flux(temperature, bottom):
        return jnp.array([(temperature[0] / 300.0)**4, -jnp.inf])

    result = solve_rce(np.array([1.0]), np.array([0.5, 2.0]), np.array([300.0]),
                       300.0 * 2**0.25, 1.0, flux, adiabatic_gradient=0.25,
                       convective_mask_initial=np.array([True]))
    assert not result.converged
    assert result.status == "nonfinite_residual"


def test_float32_cannot_converge_with_inconsistent_physical_gradients():
    jax.config.update("jax_enable_x64", False)

    def flux(temperature, bottom):
        return jnp.array([(temperature[0] / 301.0)**4, 0.1])

    result = solve_rce(np.array([1.0]), np.array([0.5, np.exp(0.03125)]),
                       np.array([301.0]), 301.0 * np.exp((0.25 + 5e-6) * 0.03125),
                       1.0, flux, adiabatic_gradient=0.25,
                       convective_mask_initial=np.array([True]))
    if result.converged:
        assert np.max(np.abs(result.gradient_residual)) <= 1e-6
        assert np.max(np.abs(result.flux_residual)) <= 1.001e-3
    else:
        assert result.status in ("inconsistent_residual", "line_search_failed")


def test_initial_domain_is_checked_in_callback_precision():
    jax.config.update("jax_enable_x64", False)

    def flux(temperature, bottom):
        raise AssertionError("Radiation must not be evaluated outside its domain.")

    with pytest.raises(ValueError, match="valid_state domain in JAX precision"):
        _toy_solve(temperature_initial=[5000.0003, 5000.0003],
                    bottom_temperature_initial=5000.0003, radiative_flux=flux,
                    valid_state=lambda t, tb: np.all(t.astype(float) >= 5000.0002))


def _gray_rce(nlayer, initial_scale=1.0, nangle=4):
    sigma = 5.670374419e-5
    boundaries = jnp.geomspace(0.01, 100.0, nlayer + 1)
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    dtau = jnp.diff(100.0 * (boundaries / 100.0)**2)
    fraction = (pressure - boundaries[:-1]) / jnp.diff(boundaries)
    mus, weights = initialize_gaussian_quadrature(nangle)

    def flux(temperature, bottom):
        tb = reconstruct_boundary_temperature(pressure, boundaries, temperature, bottom)
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, sigma * tb**4, mus, weights,
            source_center=sigma * temperature**4, upper_fraction=fraction,
        )
        return up - down

    initial = initial_scale * 150.0 * (0.5 + 75.0 * (pressure / 100.0)**2)**0.25
    result = solve_rce(pressure, boundaries, initial, initial_scale * 450.0,
                       sigma * 150.0**4, flux, flux_atol=1e-5, flux_rtol=1e-8)
    return pressure, boundaries, result


def test_gray_rce_grid_refinement_and_convective_boundary():
    profiles, bottom_temperatures, rcb = [], [], []
    for nlayer in (16, 32, 64):
        pressure, boundaries, result = _gray_rce(nlayer)
        assert result.converged, result.status
        active = np.flatnonzero(result.convective_mask)
        # A single deep convective region, with no spurious upper convection.
        assert active.size > 0
        np.testing.assert_array_equal(active, np.arange(active[0], nlayer))
        assert np.min(result.convective_flux) >= -1e-5
        assert np.max(result.gradient_residual) <= 1e-6
        assert np.max(np.abs(result.gradient_residual[result.convective_mask])) <= 1e-6
        assert np.max(np.abs(result.flux_residual)) <= 1e-5 + 1e-8 * 5.670374419e-5 * 150.0**4
        rcb.append(float(boundaries[active[0] + 1]))
        bottom_temperatures.append(result.bottom_temperature)
        profiles.append(np.interp(np.log([1.0, 10.0, 100.0]),
                                  np.log(np.append(pressure, boundaries[-1])),
                                  np.append(result.temperature, result.bottom_temperature)))
    change_coarse = np.max(np.abs(profiles[1] - profiles[0]))
    change_fine = np.max(np.abs(profiles[2] - profiles[1]))
    assert change_fine < 0.6 * change_coarse
    assert change_fine / bottom_temperatures[-1] < 0.002
    assert abs(np.log(rcb[-1] / rcb[-2])) <= np.log(1e4) / 32.0 + 1e-12


def test_gray_rce_warm_cold_starts_and_angular_resolution():
    _, _, cold = _gray_rce(24, initial_scale=0.6, nangle=4)
    _, _, warm = _gray_rce(24, initial_scale=1.6, nangle=4)
    _, _, refined = _gray_rce(24, nangle=8)
    for result in (cold, warm, refined):
        assert result.converged, result.status
    np.testing.assert_allclose(cold.temperature, warm.temperature, rtol=1e-6)
    np.testing.assert_array_equal(cold.convective_mask, warm.convective_mask)
    np.testing.assert_allclose(cold.bottom_temperature, refined.bottom_temperature, rtol=2e-3)


@pytest.mark.parametrize("options", [
    dict(pressure_bar=[4.0, 1.0]),
    dict(pressure_boundaries_bar=[0.0, 2.0, 16.0]),
    dict(temperature_initial=[0.0, 300.0]),
    dict(internal_flux=-1.0),
    dict(adiabatic_gradient=[0.25]),
    dict(convective_mask_initial=[0, 1]),
    dict(flux_atol=0.0, flux_rtol=0.0),
    dict(max_iterations=0),
    dict(radiative_flux=lambda t, tb: jnp.zeros(2)),
])
def test_invalid_inputs_raise(options):
    with pytest.raises(ValueError):
        _toy_solve(**options)
