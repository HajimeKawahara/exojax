"""Device RCE parity, guarded failures, and reusable compiled execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from exojax.atm.rce import rce_residual, solve_rce
from exojax.atm.rce_device import (
    ColumnEvaluation,
    RceStatus,
    make_device_rce_solver,
    make_gradient_evaluator,
)
from exojax.rt.flux import (
    direct_beam_fluxes,
    reconstruct_boundary_temperature,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.rtransfer import initialize_gaussian_quadrature


def _flux(temperature, bottom, parameters):
    log_t = jnp.log(jnp.append(temperature, bottom))
    return jnp.concatenate(
        ((temperature[:1] / 300.0) ** 4, parameters["conductivity"] * jnp.diff(log_t))
    )


def _prepare(**kwargs):
    inputs = dict(
        pressure_bar=np.array([1.0, 4.0]),
        pressure_boundaries_bar=np.array([0.5, 2.0, 16.0]),
        temperature_initial=np.array([250.0, 350.0]),
        bottom_temperature_initial=400.0,
        internal_flux=lambda p: p["internal_flux"],
        evaluate_column=make_gradient_evaluator([1.0, 4.0], 16.0, _flux, 0.25),
        flux_atol=1.0e-9,
        flux_rtol=1.0e-9,
        stability_atol=1.0e-9,
    )
    inputs.update(kwargs)
    return make_device_rce_solver(**inputs)


def _parameters():
    return {"internal_flux": jnp.array(1.0), "conductivity": jnp.array([4.0, 1.0])}


@pytest.mark.parametrize("jacobian_mode", ["sequential", "jacfwd"])
@pytest.mark.parametrize("initial_mask", [[False, False], [True, True]])
def test_mixed_convection_matches_host_and_residual(jacobian_mode, initial_mask):
    parameters = _parameters()
    options = dict(
        convective_mask_initial=np.array(initial_mask), jacobian_mode=jacobian_mode
    )
    result = _prepare(**options)(parameters)
    host = solve_rce(
        [1.0, 4.0],
        [0.5, 2.0, 16.0],
        [250.0, 350.0],
        400.0,
        1.0,
        lambda t, tb: _flux(t, tb, parameters),
        0.25,
        flux_atol=1e-9,
        flux_rtol=1e-9,
        gradient_atol=1e-9,
        **options,
    )
    assert result.status == RceStatus.CONVERGED
    assert result.converged and result.physics_valid and result.domain_valid
    np.testing.assert_array_equal(result.convective_mask, [False, True])
    np.testing.assert_allclose(result.temperature, host.temperature, rtol=1e-10)
    np.testing.assert_allclose(
        result.bottom_temperature, host.bottom_temperature, rtol=1e-10
    )
    np.testing.assert_allclose(result.radiative_flux, host.radiative_flux, atol=2e-9)
    np.testing.assert_allclose(result.convective_flux, host.convective_flux, atol=2e-9)
    log_t = jnp.log(jnp.append(result.temperature, result.bottom_temperature))
    expected = rce_residual(
        log_t,
        jnp.array([1.0, 4.0]),
        16.0,
        1.0,
        lambda t, tb: _flux(t, tb, parameters),
        0.25,
        result.convective_mask,
        2e-9,
        1e-9,
    )
    np.testing.assert_allclose(result.scaled_residual, expected, atol=1e-6)
    assert np.max(np.abs(result.flux_residual)) <= 2e-9
    assert np.max(result.stability_residual) <= 1e-9


def test_temperature_dependent_gradient_and_generic_stability():
    def gradient(t, tb, p):
        return 0.2 + p["slope"] * t / 300.0

    parameters = dict(_parameters(), slope=jnp.array(0.02))
    evaluator = make_gradient_evaluator([1.0, 4.0], 16.0, _flux, gradient)
    result = _prepare(evaluate_column=evaluator)(parameters)
    host = solve_rce(
        [1.0, 4.0],
        [0.5, 2.0, 16.0],
        [250.0, 350.0],
        400.0,
        1.0,
        lambda t, tb: _flux(t, tb, parameters),
        lambda t, tb: gradient(t, tb, parameters),
        flux_atol=1e-9,
        flux_rtol=1e-9,
        gradient_atol=1e-9,
    )
    assert result.converged and host.converged
    np.testing.assert_allclose(result.temperature, host.temperature, rtol=1e-10)
    np.testing.assert_allclose(
        result.bottom_temperature, host.bottom_temperature, rtol=1e-10
    )

    # An alternate dimensionless normalization uses its matching tolerance.
    def scaled(t, tb, p):
        column = evaluator(t, tb, p)
        return column._replace(stability_excess=13.0 * column.stability_excess)

    other = _prepare(evaluate_column=scaled, stability_atol=13e-9)(parameters)
    assert other.converged
    np.testing.assert_array_equal(other.convective_mask, result.convective_mask)
    np.testing.assert_allclose(other.temperature, result.temperature, rtol=1e-10)


@pytest.mark.parametrize(
    "stellar_flux,internal_flux", [(0.0, 1e4), (2e4, 1e4), (2e4, 0.0)]
)
def test_gray_irradiated_parity(stellar_flux, internal_flux):
    sigma = 5.670374419e-5
    boundaries = jnp.geomspace(0.01, 10.0, 5)
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    dtau = jnp.array([0.2, 0.5, 1.0, 2.0])

    def flux(t, tb, stellar_flux):
        boundary_t = reconstruct_boundary_temperature(pressure, boundaries, t, tb)
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, sigma * boundary_t**4, jnp.array([0.5]), jnp.array([1.0])
        )
        return up - down - direct_beam_fluxes(0.3 * dtau, stellar_flux, 0.6)

    inputs = (pressure, boundaries, jnp.full(4, 160.0), 220.0, internal_flux)
    options = dict(flux_atol=1e-6, flux_rtol=1e-10)
    host = solve_rce(*inputs, lambda t, tb: flux(t, tb, stellar_flux), 10.0, **options)
    result = make_device_rce_solver(
        *inputs,
        make_gradient_evaluator(pressure, boundaries[-1], flux, 10.0),
        **options,
    )(stellar_flux)
    assert host.converged and result.converged
    np.testing.assert_array_equal(result.convective_mask, host.convective_mask)
    np.testing.assert_allclose(result.temperature, host.temperature, rtol=1e-10)
    np.testing.assert_allclose(result.radiative_flux, host.radiative_flux, atol=2e-6)


def test_gray_convective_parity():
    sigma = 5.670374419e-5
    boundaries = jnp.geomspace(0.01, 100.0, 17)
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    dtau = jnp.diff(100.0 * (boundaries / 100.0) ** 2)
    mus, weights = initialize_gaussian_quadrature(4)

    def flux(t, tb, p):
        boundary_t = reconstruct_boundary_temperature(pressure, boundaries, t, tb)
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau,
            sigma * boundary_t**4,
            mus,
            weights,
            source_center=sigma * t**4,
            upper_fraction=(pressure - boundaries[:-1]) / jnp.diff(boundaries),
        )
        return up - down

    initial = 150.0 * (0.5 + 75.0 * (pressure / 100.0) ** 2) ** 0.25
    inputs = (pressure, boundaries, initial, 450.0, sigma * 150.0**4)
    options = dict(flux_atol=1e-5, flux_rtol=1e-8)
    host = solve_rce(*inputs, lambda t, tb: flux(t, tb, None), 2 / 7, **options)
    result = make_device_rce_solver(
        *inputs,
        make_gradient_evaluator(pressure, boundaries[-1], flux, 2 / 7),
        **options,
    )(None)
    assert result.converged and host.converged
    assert np.any(result.convective_mask)
    np.testing.assert_array_equal(result.convective_mask, host.convective_mask)
    np.testing.assert_allclose(result.temperature, host.temperature, rtol=1e-8)
    np.testing.assert_allclose(result.radiative_flux, host.radiative_flux, atol=1e-3)


@pytest.mark.parametrize(
    "option,status",
    [
        ({"max_iterations": 1}, RceStatus.MAX_ITERATIONS),
        ({"max_active_set_iterations": 1}, RceStatus.MAX_ACTIVE_SET_ITERATIONS),
        (
            {"valid_temperature": lambda t, tb, p: jnp.array(False)},
            RceStatus.INVALID_INITIAL_STATE,
        ),
    ],
)
def test_failure_statuses_and_last_accepted_state(option, status):
    result = _prepare(**option)(_parameters())
    assert not result.converged
    assert result.status == status
    assert np.all(np.isfinite(result.temperature))
    if status == RceStatus.INVALID_INITIAL_STATE:
        assert not result.domain_valid and not result.physics_valid
        assert result.physics_status == -1
        assert result.iterations == 0
        np.testing.assert_allclose(result.temperature, [250.0, 350.0], rtol=1e-14)


@pytest.mark.parametrize("flux", [-1.0, np.inf, np.nan])
def test_dynamic_internal_flux_is_checked(flux):
    result = _prepare()(dict(_parameters(), internal_flux=jnp.asarray(flux)))
    assert result.status == RceStatus.INVALID_INTERNAL_FLUX
    assert not result.converged and not result.physics_valid
    assert result.physics_status == -1
    assert result.iterations == 0


def test_dynamic_zero_internal_flux_requires_positive_tolerance():
    result = _prepare(flux_atol=0.0)(dict(_parameters(), internal_flux=jnp.array(0.0)))
    assert result.status == RceStatus.INVALID_INTERNAL_FLUX


@pytest.mark.parametrize(
    "kind,status",
    [
        ("singular", RceStatus.SINGULAR_JACOBIAN),
        ("nonfinite", RceStatus.NONFINITE_RESIDUAL),
        ("physics", RceStatus.INVALID_PHYSICS),
        ("negative_gradient", RceStatus.INVALID_PHYSICS),
    ],
)
def test_invalid_columns(kind, status):
    evaluate = make_gradient_evaluator(
        [1.0, 4.0], 16.0, _flux, -0.25 if kind == "negative_gradient" else 0.25
    )

    def column(t, tb, p):
        value = evaluate(t, tb, p)
        if kind == "singular":
            return value._replace(net_flux=jnp.zeros(3))
        if kind == "nonfinite":
            return value._replace(net_flux=jnp.array([1.0, 1.0, jnp.nan]))
        if kind == "physics":
            return value._replace(physics_valid=False, physics_status=71)
        return value

    result = _prepare(evaluate_column=column)(_parameters())
    assert result.status == status
    assert not result.converged
    assert result.physics_valid == (kind == "singular")
    if kind == "physics":
        assert result.physics_status == 71
    np.testing.assert_allclose(result.temperature, [250.0, 350.0])


def test_active_set_cycle_uses_only_valid_history_entries():
    def flux(t, tb, p):
        return jnp.array([t[0], 1.5 - jnp.log(tb / t[0])])

    solve = make_device_rce_solver(
        [1.0],
        [0.5, np.e],
        [1.0],
        np.exp(0.4),
        1.0,
        make_gradient_evaluator([1.0], np.e, flux, 0.25),
    )
    result = solve(None)
    assert result.status == RceStatus.ACTIVE_SET_CYCLE
    assert result.active_set_iterations == 3
    assert result.iterations > 0
    assert not result.converged


def test_domain_backtracking_and_domain_failure_preserve_state():
    result = _prepare(
        temperature_initial=[180.0, 200.0],
        bottom_temperature_initial=230.0,
        convective_mask_initial=np.array([False, True]),
        valid_temperature=lambda t, tb, p: jnp.all(t <= 450) & (tb <= 650),
    )(_parameters())
    assert result.converged
    assert result.backtracks > 0
    result = _prepare(
        temperature_initial=[200.0, 200.0],
        bottom_temperature_initial=200.0,
        valid_temperature=lambda t, tb, p: jnp.all(t <= 200.0),
    )(_parameters())
    assert result.status == RceStatus.DOMAIN_STEP_FAILED
    assert result.domain_valid and result.physics_valid
    assert result.backtracks == 25
    assert result.iterations == 0
    np.testing.assert_allclose(result.temperature, [200.0, 200.0])


def test_line_search_failure_returns_last_physics_diagnostics():
    base = make_gradient_evaluator([1.0, 4.0], 16.0, _flux, 0.25)

    def column(t, tb, p):
        value = base(t, tb, p)
        valid = jnp.all(t <= 200.0)
        return value._replace(
            physics_valid=valid, physics_status=jnp.where(valid, 0, 9)
        )

    result = _prepare(
        temperature_initial=[200.0, 200.0],
        bottom_temperature_initial=200.0,
        evaluate_column=column,
    )(_parameters())
    assert result.status == RceStatus.LINE_SEARCH_FAILED
    assert result.physics_valid and result.physics_status == 0
    assert result.backtracks == 25
    np.testing.assert_allclose(result.temperature, [200.0, 200.0])


def test_guarded_mixed_batch_skips_invalid_physics_and_needs_no_callback():
    base = make_gradient_evaluator([1.0, 4.0], 16.0, _flux, 0.25)

    def column(t, tb, p):
        checkify.check(p["valid"], "Invalid parameter reached column physics")
        return base(t, tb, p)

    solve = _prepare(
        evaluate_column=column, valid_temperature=lambda t, tb, p: p["valid"]
    )
    parameters = {
        "valid": jnp.array([True, False, True]),
        "internal_flux": jnp.array([1.0, 1.0, 1.1]),
        "conductivity": jnp.array([[4.0, 1.0]] * 3),
    }
    error, result = jax.jit(checkify.checkify(lambda p: jax.lax.map(solve, p)))(
        parameters
    )
    error.throw()
    np.testing.assert_array_equal(result.converged, [True, False, True])
    np.testing.assert_array_equal(
        result.status,
        [RceStatus.CONVERGED, RceStatus.INVALID_INITIAL_STATE, RceStatus.CONVERGED],
    )
    # Checkify is used only in the test. The production primal contains no callback.
    jaxpr = str(jax.make_jaxpr(_prepare())(_parameters()))
    assert "callback" not in jaxpr
    assert "while[" in jaxpr


def test_repeated_same_shape_parameters_do_not_retrace_or_change_initial_state():
    traces = []
    base = make_gradient_evaluator([1.0, 4.0], 16.0, _flux, 0.25)

    def column(t, tb, p):
        traces.append(None)
        return base(t, tb, p)

    solve = _prepare(evaluate_column=column)
    original = solve(_parameters())
    original.temperature.block_until_ready()
    count = len(traces)
    changed = solve(dict(_parameters(), internal_flux=jnp.array(1.1)))
    repeated = solve(_parameters())
    repeated.temperature.block_until_ready()
    assert count > 0 and len(traces) == count
    assert changed.converged and repeated.converged
    assert not np.allclose(original.temperature, changed.temperature)
    for a, b in zip(original, repeated):
        np.testing.assert_array_equal(a, b)


def test_float32_initial_domain_guard_uses_actual_callback_precision():
    jax.config.update("jax_enable_x64", False)
    solve = _prepare(
        temperature_initial=[5000.0003, 5000.0003],
        bottom_temperature_initial=5000.0003,
        valid_temperature=lambda t, tb, p: jnp.all(t >= jnp.float32(5000.0002)),
    )
    result = solve(_parameters())
    # This initial exp(log(T)) rounds down in float32, outside the table.
    assert result.status == RceStatus.INVALID_INITIAL_STATE
    assert result.physics_status == -1


@pytest.mark.parametrize(
    "options",
    [
        dict(pressure_bar=[4.0, 1.0]),
        dict(pressure_boundaries_bar=[0.0, 2.0, 16.0]),
        dict(temperature_initial=[0.0, 300.0]),
        dict(bottom_temperature_initial=[400.0]),
        dict(internal_flux=-1.0),
        dict(convective_mask_initial=[0, 1]),
        dict(flux_atol=0.0, flux_rtol=0.0),
        dict(stability_atol=0.0),
        dict(max_iterations=0),
        dict(max_active_set_iterations=True),
        dict(max_backtracks=2.5),
        dict(jacobian_mode="invalid"),
    ],
)
def test_static_input_validation(options):
    with pytest.raises(ValueError):
        _prepare(**options)


@pytest.mark.parametrize(
    "column",
    [
        ColumnEvaluation(jnp.zeros(2), jnp.zeros(2), True, 0),
        ColumnEvaluation(jnp.zeros(3), jnp.zeros(3), True, 0),
        ColumnEvaluation(jnp.zeros(3), jnp.zeros(2), 1, 0),
        ColumnEvaluation(jnp.zeros(3), jnp.zeros(2), True, 0.0),
    ],
)
def test_callback_contract_errors_raise_at_trace(column):
    with pytest.raises(ValueError):
        _prepare(evaluate_column=lambda t, tb, p: column)(_parameters())


@pytest.mark.parametrize("x64,tolerance", [(False, 1e-50), (True, 1e-320)])
def test_subnormal_tolerances_cannot_produce_false_convergence(x64, tolerance):
    jax.config.update("jax_enable_x64", x64)
    with pytest.raises(ValueError, match="representable"):
        _prepare(stability_atol=tolerance)


@pytest.mark.parametrize("activate", [False, True])
def test_overflowed_scaled_residual_is_not_convergence(activate):
    def column(t, tb, p):
        return ColumnEvaluation(
            jnp.array([1.0, 1.0 if activate else 1e300]),
            jnp.array([1e300 if activate else 0.0]),
            True,
            0,
        )

    solve = make_device_rce_solver(
        [1.0],
        [0.5, 2.0],
        [300.0],
        400.0,
        1.0,
        column,
        flux_atol=1e-10,
        flux_rtol=0.0,
        stability_atol=1e-10,
    )
    result = solve(None)
    assert result.status == RceStatus.NONFINITE_RESIDUAL
    assert not result.converged
    assert result.active_set_iterations == int(activate)


def test_final_total_flux_cancellation_is_not_convergence():
    # Reduced active equations are exactly satisfied, but adding huge opposing
    # radiative/convective fluxes loses the unit internal flux in float64.
    def column(t, tb, p):
        return ColumnEvaluation(jnp.array([1.0, -1e20]), jnp.zeros(1), True, 0)

    solve = make_device_rce_solver(
        [1.0],
        [0.5, 2.0],
        [300.0],
        400.0,
        1.0,
        column,
        convective_mask_initial=np.array([True]),
    )
    result = solve(None)
    assert result.status == RceStatus.INCONSISTENT_RESIDUAL
    assert not result.converged
    assert abs(result.flux_residual[-1]) > 1e-3


def test_nonfinite_jacobian_is_distinct_from_invalid_physics():
    @jax.custom_jvp
    def bad_flux(t):
        return jnp.zeros(3)

    @bad_flux.defjvp
    def bad_flux_jvp(primals, tangents):
        return bad_flux(primals[0]), jnp.full(3, jnp.nan)

    result = _prepare(
        evaluate_column=make_gradient_evaluator(
            [1.0, 4.0], 16.0, lambda t, tb, p: bad_flux(t), 0.25
        )
    )(_parameters())
    assert result.status == RceStatus.NONFINITE_JACOBIAN
    assert result.physics_valid
    assert result.iterations == 0


def test_domain_guard_also_isolates_rejected_trial_evaluations():
    base = make_gradient_evaluator([1.0, 4.0], 16.0, _flux, 0.25)

    def column(t, tb, p):
        checkify.check(jnp.all(t <= 200.0), "Trial outside temperature domain")
        return base(t, tb, p)

    solve = _prepare(
        temperature_initial=[200.0, 200.0],
        bottom_temperature_initial=200.0,
        evaluate_column=column,
        valid_temperature=lambda t, tb, p: jnp.all(t <= 200.0),
    )
    error, result = jax.jit(checkify.checkify(solve))(_parameters())
    error.throw()
    assert result.status == RceStatus.DOMAIN_STEP_FAILED
    assert result.backtracks == 25
