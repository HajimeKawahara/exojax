"""Device implicit AD, finite differences, and safe likelihood rejection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from exojax.atm.rce_device import ColumnEvaluation, RceStatus, make_gradient_evaluator
from exojax.atm.rce_device_implicit import (
    DerivativeStatus,
    LogProbStatus,
    make_device_implicit_rce_solver,
    make_rce_log_prob,
)


def _flux(t, tb, p):
    u = jnp.log(jnp.append(t, tb))
    return jnp.concatenate(((t[:1] / 300.0) ** 4, p["k"] * jnp.diff(u)))


def _gradient(t, tb, p):
    return 0.23 + p["slope"] * t / 300.0


def _parameters():
    return {
        "flux": jnp.array(1.0),
        "k": jnp.array([4.0, 1.0]),
        "slope": jnp.array(0.02),
    }


def _prepare(**options):
    inputs = dict(
        pressure_bar=[1.0, 4.0],
        pressure_boundaries_bar=[0.5, 2.0, 16.0],
        temperature_initial=[250.0, 350.0],
        bottom_temperature_initial=400.0,
        internal_flux=lambda p: p["flux"],
        evaluate_column=make_gradient_evaluator([1.0, 4.0], 16.0, _flux, _gradient),
        flux_atol=1e-10,
        flux_rtol=1e-10,
        stability_atol=1e-10,
    )
    inputs.update(options)
    return make_device_implicit_rce_solver(**inputs)


def _observable(solve, p):
    state = solve(p).state
    return jnp.concatenate(
        (
            state.temperature / 300.0,
            jnp.atleast_1d(state.bottom_temperature / 300.0),
            state.radiative_flux,
            state.convective_flux,
            state.stability_residual,
        )
    )


@pytest.mark.parametrize("mode", ["sequential", "jacfwd"])
def test_jvp_vjp_match_reconverged_finite_differences(mode):
    solve = _prepare(jacobian_mode=mode)
    p = _parameters()
    direction = {
        "flux": jnp.array(0.13),
        "k": jnp.array([0.07, -0.11]),
        "slope": jnp.array(0.003),
    }
    state = solve(p)
    assert state.state.converged and state.derivative_valid
    assert state.derivative_status == DerivativeStatus.VALID
    assert state.linear_residual <= 1
    np.testing.assert_array_equal(state.state.convective_mask, [False, True])
    function = jax.jit(lambda p: _observable(solve, p))
    _, tangent = jax.jit(lambda p, v: jax.jvp(function, (p,), (v,)))(p, direction)
    for h in (1e-3, 3e-4, 1e-4):
        plus = jax.tree_util.tree_map(lambda x, dx: x + h * dx, p, direction)
        minus = jax.tree_util.tree_map(lambda x, dx: x - h * dx, p, direction)
        assert solve(plus).derivative_valid and solve(minus).derivative_valid
        difference = (function(plus) - function(minus)) / (2 * h)
        np.testing.assert_allclose(tangent, difference, rtol=2e-6, atol=2e-8)

    weights = jnp.arange(1, tangent.size + 1, dtype=float)
    gradient = jax.jit(jax.grad(lambda p: jnp.dot(function(p), weights)))(p)
    projection = sum(
        jnp.vdot(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(direction)
        )
    )
    np.testing.assert_allclose(projection, jnp.dot(weights, tangent), rtol=1e-12)
    # Integer/bool diagnostics have float0 tangents; float diagnostics are zero.
    _, diagnostics_tangent = jax.jvp(solve, (p,), (direction,))
    assert diagnostics_tangent.state.status.dtype == jax.dtypes.float0
    np.testing.assert_array_equal(diagnostics_tangent.complementarity_margin, 0)
    assert diagnostics_tangent.linear_residual == 0


@pytest.mark.parametrize("mode", ["sequential", "jacfwd"])
def test_jitted_log_probability_and_gradient(mode):
    solve = _prepare(jacobian_mode=mode, invalid_derivative="zero")
    log_prob = make_rce_log_prob(
        solve,
        lambda state, p: -0.5 * ((state.bottom_temperature - 520.0) / 20.0) ** 2
        - 0.5 * p["slope"] ** 2,
    )
    p = _parameters()
    (value, diagnostics), gradient = jax.jit(
        jax.value_and_grad(log_prob, has_aux=True)
    )(p)
    assert jnp.isfinite(value) and diagnostics.status == LogProbStatus.ACCEPTED
    for h in (1e-3, 3e-4, 1e-4):
        plus, minus = dict(p, flux=p["flux"] + h), dict(p, flux=p["flux"] - h)
        finite_difference = (log_prob(plus)[0] - log_prob(minus)[0]) / (2 * h)
        np.testing.assert_allclose(gradient["flux"], finite_difference, rtol=1e-5)


def _one_layer(column, **options):
    inputs = dict(
        pressure_bar=[1.0],
        pressure_boundaries_bar=[0.5, 4.0],
        temperature_initial=[1.0],
        bottom_temperature_initial=1.0,
        internal_flux=1.0,
        evaluate_column=column,
        flux_atol=1e-10,
        flux_rtol=0.0,
        stability_atol=1e-10,
    )
    inputs.update(options)
    return make_device_implicit_rce_solver(**inputs)


def _column(t, tb, p):
    return ColumnEvaluation(jnp.array([t[0] + p, tb]), jnp.array([-0.1]), True, 0)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("singular", DerivativeStatus.SINGULAR_JACOBIAN),
        ("temperature", DerivativeStatus.NONFINITE_JACOBIAN),
        ("parameter", DerivativeStatus.NONFINITE_JACOBIAN),
        ("inactive_stability", DerivativeStatus.NONFINITE_JACOBIAN),
        ("smoothness", DerivativeStatus.NONSMOOTH_PHYSICS),
    ],
)
@pytest.mark.parametrize("policy", ["nan", "zero"])
def test_invalid_local_derivatives_are_explicit(kind, expected, policy):
    def column(t, tb, p):
        flux = jnp.array([t[0], tb])
        excess = jnp.array([-0.1])
        if kind == "singular":
            flux = jnp.ones(2)
        elif kind == "temperature":
            flux = flux.at[0].set(1 + jnp.sqrt(jnp.abs(t[0] - 1)))
        elif kind == "parameter":
            flux = flux.at[0].add(jnp.sqrt(p))
        elif kind == "inactive_stability":
            excess = excess + jnp.sqrt(p)
        return ColumnEvaluation(flux, excess, True, 0)

    solve = _one_layer(
        column,
        invalid_derivative=policy,
        local_smoothness=lambda t, tb, p: jnp.asarray(kind != "smoothness"),
    )
    result = solve(jnp.array(0.0))
    assert result.state.converged and not result.derivative_valid
    assert result.derivative_status == expected
    _, tangent = jax.jvp(
        lambda p: solve(p).state.temperature, (jnp.array(0.0),), (jnp.array(1.0),)
    )
    gradient = jax.jit(jax.grad(lambda p: solve(p).state.temperature.sum()))(0.0)
    if policy == "nan":
        assert jnp.all(jnp.isnan(tangent)) and jnp.isnan(gradient)
    else:
        np.testing.assert_array_equal(tangent, 0)
        assert gradient == 0


@pytest.mark.parametrize("active", [False, True])
def test_convective_switch_has_no_strict_sensitivity(active):
    # At p=0 both radiative and active equations hold, but neither inequality is strict.
    def column(t, tb, p):
        return ColumnEvaluation(
            jnp.array([t[0], tb - p]), jnp.array([tb - 1.0]), True, 0
        )

    solve = _one_layer(
        column, convective_mask_initial=np.array([active]), invalid_derivative="zero"
    )
    result = solve(0.0)
    assert result.state.converged and not result.derivative_valid
    assert result.derivative_status == DerivativeStatus.CONVECTIVE_SWITCH
    log_prob = make_rce_log_prob(solve, lambda state, p: -(state.bottom_temperature**2))
    (value, diagnostics), gradient = jax.jit(
        jax.value_and_grad(log_prob, has_aux=True)
    )(0.0)
    assert jnp.isneginf(value) and gradient == 0
    assert diagnostics.status == LogProbStatus.DERIVATIVE_INVALID


def test_failed_primal_skips_local_derivative_and_downstream_evaluation():
    @jax.custom_jvp
    def finite_with_forbidden_derivative(x):
        return x

    @finite_with_forbidden_derivative.defjvp
    def invalid_jvp(primals, tangents):
        checkify.check(jnp.asarray(False), "Failed primal reached local derivative")
        return primals[0], tangents[0]

    def column(t, tb, p):
        return ColumnEvaluation(
            jnp.array([finite_with_forbidden_derivative(t[0]), tb]),
            jnp.array([-0.1]),
            False,
            37,
        )

    solve = _one_layer(column, invalid_derivative="zero")

    def likelihood(state, p):
        checkify.check(state.physics_valid, "Failed primal reached likelihood")
        return -jnp.sqrt(state.temperature[0] - 2)

    log_prob = make_rce_log_prob(solve, likelihood)
    checked = checkify.checkify(jax.value_and_grad(log_prob, has_aux=True))
    error, ((value, diagnostics), gradient) = jax.jit(checked)(0.0)
    error.throw()
    assert jnp.isneginf(value) and gradient == 0
    assert diagnostics.status == LogProbStatus.PRIMAL_FAILED
    assert diagnostics.primal_status == RceStatus.INVALID_PHYSICS


def test_mixed_lax_map_guards_priors_domains_sensitivity_and_likelihood():
    def column(t, tb, p):
        checkify.check(p >= -1, "Domain-invalid point reached physics")
        return _column(t, tb, p)

    solve = _one_layer(
        column,
        invalid_derivative="zero",
        valid_temperature=lambda t, tb, p: p >= -1,
        local_smoothness=lambda t, tb, p: p != 0.0,
    )

    def log_density(state, p):
        checkify.check(p > 0, "Rejected point reached log density")
        return -0.5 * state.temperature[0] ** 2 + jnp.log(p)

    log_prob = make_rce_log_prob(
        solve,
        log_density,
        prior_valid=lambda p: p > -3,
        likelihood_valid=lambda state, p: p > 0,
    )

    def batched(p):
        return jax.lax.map(jax.value_and_grad(log_prob, has_aux=True), p)

    error, ((values, diagnostics), gradients) = jax.jit(checkify.checkify(batched))(
        jnp.array([0.1, -4.0, -2.0, 0.0, -0.1, 0.2])
    )
    error.throw()
    np.testing.assert_array_equal(
        diagnostics.status,
        [
            LogProbStatus.ACCEPTED,
            LogProbStatus.PRIOR_EXCLUDED,
            LogProbStatus.PRIMAL_FAILED,
            LogProbStatus.DERIVATIVE_INVALID,
            LogProbStatus.LIKELIHOOD_EXCLUDED,
            LogProbStatus.ACCEPTED,
        ],
    )
    assert jnp.all(jnp.isfinite(values[jnp.array([0, 5])]))
    assert jnp.all(jnp.isneginf(values[1:5]))
    np.testing.assert_array_equal(gradients[1:5], 0)
    np.testing.assert_allclose(gradients[jnp.array([0, 5])], [10.9, 5.8], rtol=1e-10)
    assert diagnostics.primal_status[1] == -1


def test_numerical_nonconvergence_is_not_prior_exclusion():
    solve = _prepare(invalid_derivative="zero", max_iterations=1)
    log_prob = make_rce_log_prob(solve, lambda state, p: -state.bottom_temperature)
    (value, diagnostics), gradient = jax.jit(
        jax.value_and_grad(log_prob, has_aux=True)
    )(_parameters())
    assert jnp.isneginf(value) and diagnostics.status == LogProbStatus.PRIMAL_FAILED
    assert diagnostics.primal_status == RceStatus.MAX_ITERATIONS
    for leaf in jax.tree_util.tree_leaves(gradient):
        np.testing.assert_array_equal(leaf, 0)


def test_no_retrace_no_host_callback_and_one_primal_solve(monkeypatch):
    import exojax.atm.rce_device_implicit as module

    original = module.make_device_rce_solver
    traces, executions = [], []

    def counting_factory(*args, **kwargs):
        primal = original(*args, **kwargs)

        def wrapped(parameters):
            traces.append(None)
            jax.debug.callback(lambda: executions.append(None), ordered=True)
            return primal(parameters)

        return wrapped

    monkeypatch.setattr(module, "make_device_rce_solver", counting_factory)
    solve = _prepare()
    objective = jax.jit(jax.value_and_grad(lambda p: solve(p).state.bottom_temperature))
    p = _parameters()
    value, _ = objective(p)
    value.block_until_ready()
    jax.effects_barrier()
    assert len(executions) == 1
    trace_count = len(traces)
    value, _ = objective(dict(p, flux=jnp.array(1.01)))
    value.block_until_ready()
    jax.effects_barrier()
    assert len(executions) == 2 and len(traces) == trace_count
    monkeypatch.undo()
    ordinary = _prepare()
    jaxpr = str(
        jax.make_jaxpr(
            jax.value_and_grad(lambda p: ordinary(p).state.bottom_temperature)
        )(p)
    )
    assert "callback" not in jaxpr and "while[" in jaxpr


def test_inaccurate_linear_certificate_rejects_without_regularizing():
    solve = _prepare(invalid_derivative="zero", linear_atol=1e-30, linear_rtol=1e-30)
    result = solve(_parameters())
    assert result.state.converged and not result.derivative_valid
    assert result.derivative_status == DerivativeStatus.INACCURATE_LINEAR_SOLVE
    assert result.linear_residual > 1


def test_relative_only_linear_certificate_allows_disconnected_parameters():
    solve = _prepare(linear_atol=0.0)
    # The conductivity of the active connection does not enter its equation.
    result = solve(_parameters())
    assert result.derivative_valid and result.linear_residual <= 1
    derivative = jax.grad(lambda p: solve(p).state.bottom_temperature)(_parameters())
    assert derivative["k"][1] == 0


def test_callback_precision_is_cast_before_shared_residual_scaling():
    def column(t, tb, p):
        return ColumnEvaluation(
            jnp.array([t[0] + p, 0.0], dtype=jnp.float32),
            jnp.array([tb - 0.75 * t[0] - 0.25], dtype=jnp.float32),
            True,
            0,
        )

    solve = _one_layer(column, convective_mask_initial=np.array([True]))
    p = jnp.float32(0.0)
    result = solve(p)
    assert result.derivative_valid and result.state.temperature.dtype == jnp.float64
    derivative = jax.jit(jax.grad(lambda p: solve(p).state.bottom_temperature))(p)
    np.testing.assert_allclose(derivative, -0.75, rtol=1e-7)


def test_nonfinite_linear_response_is_rejected():
    def column(t, tb, p):
        values = jnp.append(t, tb)
        flux = 1 + 1e-100 * (values - 1) + 1e300 * p
        return ColumnEvaluation(flux, jnp.array([-0.1]), True, 0)

    solve = _one_layer(column, flux_atol=1.0, invalid_derivative="zero")
    result = solve(0.0)
    assert result.state.converged and not result.derivative_valid
    assert result.derivative_status == DerivativeStatus.NONFINITE_SENSITIVITY
    assert jax.grad(lambda p: solve(p).state.temperature.sum())(0.0) == 0


@pytest.mark.parametrize(
    "callback",
    [
        lambda state, p: jnp.sqrt(p - 1),
        lambda state, p: 1.0 / p,
    ],
)
def test_nonfinite_log_density_rejects_before_differentiating(callback):
    solve = _one_layer(_column, invalid_derivative="zero")
    log_prob = make_rce_log_prob(solve, callback)
    (value, diagnostics), gradient = jax.jit(
        jax.value_and_grad(log_prob, has_aux=True)
    )(0.0)
    assert jnp.isneginf(value) and gradient == 0
    assert diagnostics.status == LogProbStatus.NONFINITE_LOG_PROB


@pytest.mark.parametrize(
    "callback",
    [
        lambda state, p: jnp.ones(2),
        lambda state, p: jnp.array(1),
    ],
)
def test_log_density_requires_scalar_floating_value(callback):
    solve = _one_layer(_column, invalid_derivative="zero")
    with pytest.raises(ValueError, match="scalar floating-point"):
        make_rce_log_prob(solve, callback)(0.0)


@pytest.mark.parametrize(
    "options",
    [
        {"switch_margin": 0.5},
        {"switch_margin": np.inf},
        {"linear_atol": -1},
        {"linear_rtol": np.nan},
        {"linear_atol": 0, "linear_rtol": 0},
        {"linear_atol": 1e-320},
        {"invalid_derivative": "approximate"},
        {"local_smoothness": True},
    ],
)
def test_static_validation(options):
    with pytest.raises((ValueError, TypeError)):
        _prepare(**options)


@pytest.mark.parametrize(
    "parameters", [None, {}, {"a": jnp.array([], dtype=float)}, 1, True, 1 + 0j]
)
def test_parameter_contract(parameters):
    solve = _one_layer(_column)
    with pytest.raises(TypeError):
        solve(parameters)


def test_strict_solver_cannot_be_used_as_guarded_likelihood():
    with pytest.raises(ValueError, match="invalid_derivative='zero'"):
        make_rce_log_prob(_prepare(), lambda state, p: -state.bottom_temperature)
