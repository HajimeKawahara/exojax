"""Implicit equilibrium derivatives against an analytic mixed-convection column."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce import solve_rce
from exojax.atm.rce_implicit import make_implicit_rce_solver


PRESSURE = np.array([0.1, 0.4])
BOUNDARIES = np.array([0.05, 0.2, 1.0])
INITIAL = np.array([155.0, 225.0])
BETA = 0.2
OPTIONS = dict(
    convective_mask_initial=np.array([False, True]),
    flux_atol=1.0e-10,
    flux_rtol=0.0,
    gradient_atol=1.0e-10,
)


def _params(vector):
    return {"forcing": vector[:2], "convection": {"offset": vector[2]}}


def _internal(params):
    return 100.0 * params["forcing"][1]


def _flux(temperature, bottom, params):
    z0, z1, zb = (jnp.append(temperature, bottom) / 200.0) ** 4
    stellar = params["forcing"][0]
    return 100.0 * jnp.array([
        0.6 * z0 + 0.2 * z1 + 0.2 * zb - stellar,
        -0.3 * z0 + 0.2 * z1 + 0.3 * zb - 0.8 * stellar,
        0.1 * zb - 0.2 * z1 - 0.1 * stellar,
    ])


def _gradient(temperature, bottom, params):
    # Algebraic feedback keeps an exact solution while exercising d gradient/dT.
    measured = jnp.log(bottom / temperature[-1]) / jnp.log(2.5)
    return jnp.array([0.25, params["convection"]["offset"] + BETA * measured])


def _make_solver(**overrides):
    settings = dict(
        pressure_bar=PRESSURE,
        pressure_boundaries_bar=BOUNDARIES,
        temperature_initial=INITIAL,
        bottom_temperature_initial=245.0,
        internal_flux=_internal,
        radiative_flux=_flux,
        neutral_gradient=_gradient,
        **OPTIONS,
    )
    settings.update(overrides)
    return make_implicit_rce_solver(**settings)


def _temperatures(result):
    return jnp.append(result.temperature, result.bottom_temperature)


def _analytic(vector):
    """Solve two linear equations in T**4 after imposing the lower gradient."""
    stellar, internal, offset = np.asarray(vector)
    q = 2.5 ** (4.0 * offset / (1.0 - BETA))
    coefficient = 0.2 + 0.2 * q
    denominator = 0.3 + 0.4 * q
    z1 = (1.5 * internal + 1.3 * stellar) / denominator
    z0 = (internal + stellar - coefficient * z1) / 0.6
    z = np.array([z0, z1, q * z1])
    temperature = 200.0 * z**0.25
    jacobian_z = np.empty((3, 3))
    for column, (ds, dh, da) in enumerate(np.eye(3)):
        dq = 4.0 * np.log(2.5) * q * da / (1.0 - BETA)
        dz1 = (1.5 * dh + 1.3 * ds - 0.4 * z1 * dq) / denominator
        dz0 = (dh + ds - coefficient * dz1 - 0.2 * z1 * dq) / 0.6
        jacobian_z[:, column] = [dz0, dz1, q * dz1 + z1 * dq]
    return temperature, temperature[:, None] * jacobian_z / (4.0 * z[:, None])


def test_forward_recomputes_parameters_and_matches_normal_solve():
    solve = jax.jit(_make_solver())
    outputs = []
    for vector in [jnp.array([1.0, 0.05, 0.16]), jnp.array([1.08, 0.07, 0.15])]:
        params = _params(vector)
        result = solve(params)
        normal = solve_rce(
            PRESSURE, BOUNDARIES, INITIAL, 245.0, float(_internal(params)),
            lambda t, tb: _flux(t, tb, params),
            neutral_gradient=lambda t, tb: _gradient(t, tb, params), **OPTIONS,
        )
        assert result.converged and normal.converged
        np.testing.assert_array_equal(result.convective_mask, normal.convective_mask)
        np.testing.assert_array_equal(result.convective_mask, [False, True])
        expected, _ = _analytic(vector)
        np.testing.assert_allclose(_temperatures(result), expected, rtol=2e-10)
        np.testing.assert_allclose(result.temperature, normal.temperature, rtol=2e-10)
        np.testing.assert_allclose(result.bottom_temperature, normal.bottom_temperature, rtol=2e-10)
        outputs.append(np.asarray(_temperatures(result)))
    assert np.max(np.abs(outputs[0] - outputs[1])) > 1.0


@pytest.mark.parametrize("jacobian_mode", ["sequential", "jacfwd"])
def test_jvp_vjp_and_jacobians_match_analytic_solution_and_finite_difference(jacobian_mode):
    solve = _make_solver(jacobian_mode=jacobian_mode)
    vector = jnp.array([1.0, 0.05, 0.16])

    def state(values):
        return _temperatures(solve(_params(values)))

    expected_temperature, expected_jacobian = _analytic(vector)
    direction = jnp.array([0.3, -0.2, 0.1])
    cotangent = jnp.array([0.7, -0.2, 1.3])
    primal, tangent = jax.jit(lambda x, v: jax.jvp(state, (x,), (v,)))(vector, direction)
    np.testing.assert_allclose(primal, expected_temperature, rtol=2e-10)
    np.testing.assert_allclose(tangent, expected_jacobian @ direction, rtol=2e-8, atol=1e-7)
    _, pullback = jax.vjp(state, vector)
    np.testing.assert_allclose(pullback(cotangent)[0], cotangent @ expected_jacobian,
                               rtol=2e-8, atol=1e-7)
    for transform in [jax.jacfwd, jax.jacrev]:
        actual = jax.jit(transform(state))(vector)
        np.testing.assert_allclose(actual, expected_jacobian, rtol=2e-8, atol=1e-7)

    tree_gradient = jax.grad(lambda p: jnp.dot(cotangent, _temperatures(solve(p))))(_params(vector))
    expected_vjp = np.asarray(cotangent) @ expected_jacobian
    np.testing.assert_allclose(tree_gradient["forcing"], expected_vjp[:2], rtol=2e-8)
    np.testing.assert_allclose(tree_gradient["convection"]["offset"], expected_vjp[2], rtol=2e-8)

    step = 1.0e-5
    differences = []
    for basis in np.eye(3):
        plus, minus = [solve(_params(vector + sign * step * basis)) for sign in [1, -1]]
        assert plus.converged and minus.converged
        np.testing.assert_array_equal(plus.convective_mask, [False, True])
        np.testing.assert_array_equal(minus.convective_mask, [False, True])
        differences.append((_temperatures(plus) - _temperatures(minus)) / (2.0 * step))
    np.testing.assert_allclose(np.column_stack(differences), expected_jacobian,
                               rtol=2e-7, atol=1e-6)


@pytest.mark.parametrize("gradient", [0.4, np.array([0.4])])
def test_constant_internal_flux_and_gradient(gradient):
    solve = make_implicit_rce_solver(
        np.array([0.5]), np.array([0.25, 1.0]), np.array([190.0]), 220.0,
        3.0, lambda t, tb, p: jnp.append(t, tb) - p, gradient,
        flux_atol=1e-10, flux_rtol=0.0,
    )
    params = jnp.array([200.0, 230.0])
    result = solve(params)
    assert result.converged
    np.testing.assert_array_equal(result.convective_mask, [False])
    np.testing.assert_allclose(_temperatures(result), params + 3.0, rtol=1e-11)
    np.testing.assert_allclose(jax.jacrev(lambda p: _temperatures(solve(p)))(params),
                               np.eye(2), atol=1e-10)


@pytest.mark.parametrize("failure", ["iterations", "invalid_initial_domain"])
def test_failed_forward_solve_has_no_equilibrium_derivative(failure):
    options = (dict(max_iterations=1) if failure == "iterations" else
               dict(valid_state=lambda t, tb, p: float(p["forcing"][0]) < 0.0))
    solve = _make_solver(**options)
    vector = jnp.array([1.0, 0.05, 0.16])
    result = jax.jit(solve)(_params(vector))
    assert not result.converged
    assert np.all(np.isnan(_temperatures(result)))
    state = lambda v: _temperatures(solve(_params(v)))
    _, tangent = jax.jvp(state, (vector,), (jnp.ones(3),))
    assert np.all(np.isnan(tangent))
    assert np.all(np.isnan(jax.grad(lambda v: jnp.sum(state(v)))(vector)))


def _switch_solver(active):
    def flux(temperature, bottom, params):
        departure = jnp.log(bottom / (temperature[0] * 2.0**0.25))
        return jnp.array([(temperature[0] / 200.0)**4, 1.0 + params[0] + departure])

    return make_implicit_rce_solver(
        np.array([0.5]), np.array([0.25, 1.0]), np.array([200.0]), 200.0 * 2.0**0.25,
        1.0, flux, 0.25, convective_mask_initial=np.array([active]),
        flux_atol=1e-10, flux_rtol=0.0, gradient_atol=1e-10,
    )


@pytest.mark.parametrize("active", [False, True])
def test_switching_boundary_keeps_primal_but_rejects_tangents(active):
    solve = _switch_solver(active)
    params = jnp.zeros(1)
    result = solve(params)
    assert result.converged
    assert np.all(np.isfinite(_temperatures(result)))
    np.testing.assert_array_equal(result.convective_mask, [active])
    state = lambda p: _temperatures(solve(p))
    _, tangent = jax.jvp(state, (params,), (jnp.ones(1),))
    assert np.all(np.isnan(tangent))
    assert np.all(np.isnan(jax.grad(lambda p: jnp.sum(state(p)))(params)))


def test_each_side_of_switch_has_its_own_smooth_derivative():
    solve = _switch_solver(True)
    for value, active in [(-0.01, True), (0.01, False)]:
        params = jnp.array([value])
        result = solve(params)
        assert result.converged
        np.testing.assert_array_equal(result.convective_mask, [active])
        derivative = jax.jacfwd(lambda p: _temperatures(solve(p)))(params)[:, 0]
        expected = np.array([0.0, 0.0 if active else -float(result.bottom_temperature)])
        np.testing.assert_allclose(derivative, expected, atol=1e-8)


def test_singular_jacobian_cannot_return_finite_equilibrium_derivative():
    solve = make_implicit_rce_solver(
        np.array([0.5]), np.array([0.25, 1.0]), np.array([200.0]), 200.0,
        1.0, lambda t, tb, p: jnp.repeat((t[0] / 200.0)**4 - p[0], 2), 0.25,
        flux_atol=1e-10, flux_rtol=0.0,
    )
    params = jnp.zeros(1)
    result = solve(params)
    assert result.converged
    assert np.all(np.isfinite(_temperatures(result)))
    derivative = jax.jacfwd(lambda p: _temperatures(solve(p)))(params)
    assert not np.all(np.isfinite(derivative))
    gradient = jax.grad(lambda p: jnp.sum(_temperatures(solve(p))))(params)
    assert not np.all(np.isfinite(gradient))


def test_bottom_initial_temperature_must_be_scalar():
    with pytest.raises(ValueError, match="scalar"):
        _make_solver(bottom_temperature_initial=np.array([245.0]))


@pytest.mark.parametrize("params", [jnp.array([1]), {"flag": True}, np.array([1.0j])])
def test_parameters_must_have_real_floating_leaves(params):
    with pytest.raises((TypeError, ValueError)):
        _make_solver()(params)
