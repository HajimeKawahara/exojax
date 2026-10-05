"""Prepared RCE primal solves with device-side active-set Newton iteration.

The host solver in :mod:`exojax.atm.rce` remains the reference implementation.
This module does not provide implicit derivatives of the converged solution.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np


class RceStatus(IntEnum):
    """Device result codes; compare the returned scalar with these members."""

    RUNNING = -1
    CONVERGED = 0
    INVALID_INITIAL_STATE = 1
    INVALID_INTERNAL_FLUX = 2
    INVALID_PHYSICS = 3
    NONFINITE_RESIDUAL = 4
    NONFINITE_JACOBIAN = 5
    SINGULAR_JACOBIAN = 6
    NONFINITE_STEP = 7
    LINE_SEARCH_FAILED = 8
    DOMAIN_STEP_FAILED = 9
    MAX_ITERATIONS = 10
    ACTIVE_SET_CYCLE = 11
    MAX_ACTIVE_SET_ITERATIONS = 12
    INCONSISTENT_RESIDUAL = 13


class ColumnEvaluation(NamedTuple):
    """Shared radiation and stability evaluation of one atmospheric column.

    ``net_flux`` has shape (N+1,), is positive upward, and is in erg/s/cm2.
    ``stability_excess`` has shape (N,), is unstable when positive, and is
    zero on neutral connections. Normalize dimensional thermodynamic quantities
    in the callback; stability_excess is dimensionless.
    ``physics_valid`` is a scalar boolean. ``physics_status`` is a scalar
    integer defined by the caller (zero for success); -1 is reserved for a
    column whose evaluation was skipped by the temperature/domain guard.
    Chemistry, opacity, and thermodynamic work can be shared in this callback.
    """

    net_flux: jax.Array
    stability_excess: jax.Array
    physics_valid: jax.Array
    physics_status: jax.Array


class DeviceRceResult(NamedTuple):
    """Last accepted state, including on failure; all fields are JAX arrays.

    ``scaled_residual`` is the fixed-mask residual divided by the tolerances.
    ``flux_residual`` includes convective flux. ``backtracks`` counts rejected
    line-search trials; ``iterations`` counts accepted Newton steps. Physics
    diagnostics refer to the returned state, never a rejected trial. Invalid
    initial states have NaN flux/stability diagnostics and physics_status=-1.
    ``converged`` describes only the primal, not derivative validity.
    """

    temperature: jax.Array
    bottom_temperature: jax.Array
    radiative_flux: jax.Array
    convective_flux: jax.Array
    convective_mask: jax.Array
    flux_residual: jax.Array
    stability_residual: jax.Array
    scaled_residual: jax.Array
    iterations: jax.Array
    active_set_iterations: jax.Array
    backtracks: jax.Array
    domain_valid: jax.Array
    physics_valid: jax.Array
    physics_status: jax.Array
    converged: jax.Array
    status: jax.Array


def make_gradient_evaluator(
    pressure_bar, bottom_pressure_bar, radiative_flux, neutral_gradient
):
    """Adapt the legacy neutral-gradient condition to ``ColumnEvaluation``.

    Radiation and gradient callbacks take ``(T, T_bottom, parameters)``. The
    gradient may instead be a constant scalar or (N,) array. Nonfinite or
    nonpositive gradients set physics_valid=False and physics_status=1.
    Their temperature dependence remains part of the Newton Jacobian.
    Construct this adapter outside JAX transformations.
    """
    pressure = np.asarray(pressure_bar, dtype=float)
    if pressure.ndim != 1 or np.ndim(bottom_pressure_bar) != 0:
        raise ValueError(
            "Expected one-dimensional centers and a scalar bottom pressure."
        )
    nodes = np.append(pressure, bottom_pressure_bar)
    if (
        nodes.ndim != 1
        or nodes.size < 2
        or not np.all(np.isfinite(nodes))
        or np.any(nodes <= 0)
        or np.any(np.diff(nodes) <= 0)
    ):
        raise ValueError("Pressure nodes must be positive and strictly increasing.")
    dlog_pressure = jnp.asarray(np.diff(np.log(nodes)))

    def evaluate(temperature, bottom_temperature, parameters):
        gradient = jnp.asarray(
            neutral_gradient(temperature, bottom_temperature, parameters)
            if callable(neutral_gradient)
            else neutral_gradient
        )
        if gradient.shape not in ((), (nodes.size - 1,)):
            raise ValueError("neutral_gradient must be scalar or have shape (N,).")
        valid = jnp.all(jnp.isfinite(gradient) & (gradient > 0))
        excess = (
            jnp.diff(jnp.log(jnp.append(temperature, bottom_temperature)))
            / dlog_pressure
            - gradient
        )
        return ColumnEvaluation(
            radiative_flux(temperature, bottom_temperature, parameters),
            excess,
            valid,
            jnp.where(valid, 0, 1),
        )

    return evaluate


def _scaled_residual(column, mask, internal_flux, flux_tolerance, stability_atol):
    flux_error = (column.net_flux - internal_flux) / flux_tolerance
    return jnp.concatenate(
        (
            flux_error[:1],
            jnp.where(mask, column.stability_excess / stability_atol, flux_error[1:]),
        )
    )


def _dense_jacobian(residual, log_temperature, mode):
    """Share the primal Jacobian construction with future implicit AD."""
    if mode == "jacfwd":
        return jax.jacfwd(residual)(log_temperature)
    columns = jax.lax.map(
        lambda tangent: jax.jvp(residual, (log_temperature,), (tangent,))[1],
        jnp.eye(log_temperature.size, dtype=log_temperature.dtype),
    )
    return columns.T


class _State(NamedTuple):
    log_t: jax.Array
    mask: jax.Array
    column: ColumnEvaluation
    residual: jax.Array
    status: jax.Array
    iterations: jax.Array
    active_iterations: jax.Array
    backtracks: jax.Array


def make_device_rce_solver(
    pressure_bar,
    pressure_boundaries_bar,
    temperature_initial,
    bottom_temperature_initial,
    internal_flux,
    evaluate_column,
    *,
    valid_temperature=None,
    convective_mask_initial=None,
    flux_atol=1.0e-3,
    flux_rtol=1.0e-6,
    stability_atol=1.0e-6,
    max_iterations=50,
    max_active_set_iterations=30,
    max_backtracks=25,
    jacobian_mode="sequential",
):
    """Prepare a reusable, JIT-compiled ``solve(parameters)`` primal solver.

    Construct once outside JAX transformations. Fixed inputs and numerical
    options follow ``solve_rce``; stability_atol replaces gradient_atol and
    applies to the caller-defined stability excess. ``evaluate_column`` takes
    ``(T, T_bottom, parameters)`` and returns ``ColumnEvaluation``. All varying
    physical inputs must be explicit leaves of the parameters PyTree, including
    any dependencies of ``internal_flux(parameters)`` (or use a fixed scalar).
    The flux tolerance remains flux_atol + flux_rtol * abs(internal_flux).

    ``valid_temperature(T, T_bottom, parameters)`` is an optional, cheap scalar
    boolean predicate evaluated before column physics. It should include fixed
    pressure/table validity as needed. Invalid trial states are shortened;
    invalid initial states return a status without running column physics.
    Physics callbacks must be pure, JAX-compatible, and locally differentiable
    at valid states. Shape/type mistakes raise on first tracing; runtime
    numerical/physical failures return integer RceStatus codes. Enable JAX x64
    before preparing precision-sensitive columns.

    Newton, line-search, and active-set loops run entirely on device. The default
    Jacobian uses sequential JVPs; ``jacfwd`` explicitly opts into batched
    tangents. No previous solve is cached as an initial state. For batches with
    invalid states, use ``jax.lax.map(solve, parameters_batch)``: unrestricted
    ``vmap`` can execute both branches of domain guards and is not supported.
    This primal interface does not supply equilibrium sensitivities; implicit
    differentiation is a separate implementation stage.
    """
    pressure = np.asarray(pressure_bar, dtype=float)
    boundaries = np.asarray(pressure_boundaries_bar, dtype=float)
    initial = np.asarray(temperature_initial, dtype=float)
    if pressure.ndim != 1 or pressure.size == 0:
        raise ValueError("pressure_bar must be a nonempty one-dimensional array.")
    nlayer = pressure.size
    if boundaries.shape != (nlayer + 1,) or initial.shape != (nlayer,):
        raise ValueError("Expected N center temperatures and N+1 pressure boundaries.")
    if (
        not np.all(np.isfinite(pressure))
        or not np.all(np.isfinite(boundaries))
        or np.any(boundaries <= 0)
        or np.any(np.diff(boundaries) <= 0)
        or np.any(pressure <= boundaries[:-1])
        or np.any(pressure >= boundaries[1:])
    ):
        raise ValueError("Pressures must increase, with every center inside its layer.")
    if np.ndim(bottom_temperature_initial) != 0:
        raise ValueError("bottom_temperature_initial must be scalar.")
    temperatures = np.append(initial, bottom_temperature_initial)
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("Initial temperatures must be finite and positive.")
    for name, value, allow_zero in (
        ("flux_atol", flux_atol, True),
        ("flux_rtol", flux_rtol, True),
        ("stability_atol", stability_atol, False),
    ):
        if (
            np.ndim(value) != 0
            or not np.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(f"Invalid {name}.")
    if flux_atol == 0 and flux_rtol == 0:
        raise ValueError("The combined flux tolerance must be positive.")
    if not callable(internal_flux):
        if (
            np.ndim(internal_flux) != 0
            or not np.isfinite(internal_flux)
            or internal_flux < 0
        ):
            raise ValueError("internal_flux must be a finite, nonnegative scalar.")
        tolerance = flux_atol + flux_rtol * abs(internal_flux)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("The combined flux tolerance must be positive and finite.")
    for name, value in (
        ("max_iterations", max_iterations),
        ("max_active_set_iterations", max_active_set_iterations),
        ("max_backtracks", max_backtracks),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(jacobian_mode, str) or jacobian_mode not in (
        "sequential",
        "jacfwd",
    ):
        raise ValueError("jacobian_mode must be 'sequential' or 'jacfwd'.")
    mask = (
        np.zeros(nlayer, dtype=bool)
        if convective_mask_initial is None
        else np.asarray(convective_mask_initial)
    )
    if mask.shape != (nlayer,) or mask.dtype != np.dtype(bool):
        raise ValueError(
            "convective_mask_initial must be a boolean array of shape (N,)."
        )
    if not callable(evaluate_column) or (
        valid_temperature is not None and not callable(valid_temperature)
    ):
        raise TypeError("Column evaluation and temperature guard must be callable.")
    log_initial = jnp.log(jnp.asarray(temperatures))
    mask_initial = jnp.array(mask, copy=True)
    dtype = log_initial.dtype
    # Subnormal tolerances can be flushed to zero by accelerator arithmetic.
    for name, value in (
        ("flux_atol", flux_atol),
        ("flux_rtol", flux_rtol),
        ("stability_atol", stability_atol),
    ):
        if value > 0 and (value < np.finfo(dtype).tiny or value > np.finfo(dtype).max):
            raise ValueError(
                f"{name} must be representable as a normal value in {dtype}."
            )
    flux_atol, flux_rtol, stability_atol = map(
        float, (flux_atol, flux_rtol, stability_atol)
    )
    if not callable(internal_flux):
        internal_flux = float(internal_flux)
    zero = jnp.asarray(0, dtype=jnp.int32)

    def domain(log_t, parameters):
        values = jnp.exp(log_t)
        valid = jnp.all(jnp.isfinite(values) & (values > 0))
        if valid_temperature is not None:
            extra = jnp.asarray(valid_temperature(values[:-1], values[-1], parameters))
            if extra.shape != () or extra.dtype != jnp.bool_:
                raise ValueError("valid_temperature must return a scalar boolean.")
            valid = valid & extra
        return valid

    def evaluate_values(values, parameters):
        column = evaluate_column(values[:-1], values[-1], parameters)
        flux = jnp.asarray(column.net_flux, dtype=dtype)
        excess = jnp.asarray(column.stability_excess, dtype=dtype)
        valid = jnp.asarray(column.physics_valid)
        status = jnp.asarray(column.physics_status)
        if flux.shape != (nlayer + 1,) or excess.shape != (nlayer,):
            raise ValueError("Column flux/stability shapes must be (N+1,)/(N,).")
        if valid.shape != () or valid.dtype != jnp.bool_:
            raise ValueError("physics_valid must be a scalar boolean.")
        if status.shape != () or not jnp.issubdtype(status.dtype, jnp.integer):
            raise ValueError("physics_status must be a scalar integer.")
        return ColumnEvaluation(flux, excess, valid, status.astype(jnp.int32))

    def evaluate(log_t, parameters):
        return evaluate_values(jnp.exp(log_t), parameters)

    empty = ColumnEvaluation(
        jnp.full((nlayer + 1,), jnp.nan, dtype=dtype),
        jnp.full((nlayer,), jnp.nan, dtype=dtype),
        jnp.asarray(False),
        zero - 1,
    )

    def column_status(column):
        finite = jnp.all(jnp.isfinite(column.net_flux)) & jnp.all(
            jnp.isfinite(column.stability_excess)
        )
        return jnp.where(
            ~finite,
            RceStatus.NONFINITE_RESIDUAL,
            jnp.where(
                column.physics_valid, RceStatus.RUNNING, RceStatus.INVALID_PHYSICS
            ),
        ).astype(jnp.int32)

    @jax.jit
    def solve(parameters):
        flux = jnp.asarray(
            internal_flux(parameters) if callable(internal_flux) else internal_flux,
            dtype=dtype,
        )
        if flux.shape != ():
            raise ValueError("internal_flux must return a scalar.")
        flux_tolerance = flux_atol + flux_rtol * jnp.abs(flux)
        valid_flux = (
            jnp.isfinite(flux)
            & (flux >= 0)
            & jnp.isfinite(flux_tolerance)
            & (flux_tolerance >= np.finfo(dtype).tiny)
        )
        valid_initial = domain(log_initial, parameters)
        column = jax.lax.cond(
            valid_initial & valid_flux,
            lambda: evaluate(log_initial, parameters),
            lambda: empty,
        )
        status = jnp.where(
            ~valid_flux,
            RceStatus.INVALID_INTERNAL_FLUX,
            jnp.where(
                ~valid_initial, RceStatus.INVALID_INITIAL_STATE, column_status(column)
            ),
        ).astype(jnp.int32)

        def residual(log_t, active):
            return _scaled_residual(
                evaluate(log_t, parameters),
                active,
                flux,
                flux_tolerance,
                stability_atol,
            )

        state = _State(
            log_initial,
            mask_initial,
            column,
            _scaled_residual(
                column, mask_initial, flux, flux_tolerance, stability_atol
            ),
            status,
            zero,
            zero,
            zero,
        )
        state = state._replace(
            status=jnp.where(
                (state.status == RceStatus.RUNNING)
                & ~jnp.all(jnp.isfinite(state.residual)),
                RceStatus.NONFINITE_RESIDUAL,
                state.status,
            )
        )
        history = jnp.zeros((max_active_set_iterations, nlayer), dtype=bool)

        def newton_step(state):
            matrix = _dense_jacobian(
                lambda u: residual(u, state.mask), state.log_t, jacobian_mode
            )
            finite_matrix = jnp.all(jnp.isfinite(matrix))
            # LU exposes exact singular pivots without a second factorization.
            lu, pivots = jlinalg.lu_factor(matrix)
            singular = jnp.any(jnp.diag(lu) == 0)
            step = jlinalg.lu_solve((lu, pivots), -state.residual)
            step_status = jnp.where(
                ~finite_matrix,
                RceStatus.NONFINITE_JACOBIAN,
                jnp.where(
                    singular,
                    RceStatus.SINGULAR_JACOBIAN,
                    jnp.where(
                        jnp.all(jnp.isfinite(step)),
                        RceStatus.RUNNING,
                        RceStatus.NONFINITE_STEP,
                    ),
                ),
            ).astype(jnp.int32)
            damping = 1.0 / jnp.maximum(1.0, jnp.max(jnp.abs(step)))
            norm = jnp.max(jnp.abs(state.residual))

            def search_body(carry):
                trial_index, damping, accepted, had_domain, current = carry
                log_trial = state.log_t + damping * step
                valid = domain(log_trial, parameters)
                trial_column = jax.lax.cond(
                    valid, lambda: evaluate(log_trial, parameters), lambda: empty
                )
                trial_residual = _scaled_residual(
                    trial_column, state.mask, flux, flux_tolerance, stability_atol
                )
                trial_norm = jnp.max(jnp.abs(trial_residual))
                accept = (
                    valid
                    & (column_status(trial_column) == RceStatus.RUNNING)
                    & jnp.isfinite(trial_norm)
                    & ((trial_norm <= 1) | (trial_norm < (1 - 1.0e-4 * damping) * norm))
                )
                current = jax.lax.cond(
                    accept,
                    lambda: current._replace(
                        log_t=log_trial,
                        column=trial_column,
                        residual=trial_residual,
                        iterations=current.iterations + 1,
                    ),
                    lambda: current._replace(backtracks=current.backtracks + 1),
                )
                return (
                    trial_index + 1,
                    damping * 0.5,
                    accept,
                    had_domain | valid,
                    current,
                )

            search_initial = (
                zero,
                damping,
                jnp.asarray(False),
                jnp.asarray(False),
                state,
            )
            _, _, accepted, had_domain, updated = jax.lax.while_loop(
                lambda carry: (step_status == RceStatus.RUNNING)
                & (carry[0] < max_backtracks)
                & ~carry[2],
                search_body,
                search_initial,
            )
            status = jnp.where(
                step_status != RceStatus.RUNNING,
                step_status,
                jnp.where(
                    accepted,
                    RceStatus.RUNNING,
                    jnp.where(
                        had_domain,
                        RceStatus.LINE_SEARCH_FAILED,
                        RceStatus.DOMAIN_STEP_FAILED,
                    ),
                ),
            ).astype(jnp.int32)
            return updated._replace(status=status)

        def active_body(carry):
            state, history, history_count = carry
            cycle = jnp.any(
                jnp.all(history == state.mask, axis=1)
                & (jnp.arange(max_active_set_iterations) < history_count)
            )
            history = history.at[history_count].set(state.mask)
            state = state._replace(
                active_iterations=state.active_iterations + 1,
                status=jnp.where(cycle, RceStatus.ACTIVE_SET_CYCLE, state.status),
            )
            start_iterations = state.iterations
            state = jax.lax.while_loop(
                lambda s: (s.status == RceStatus.RUNNING)
                & (s.iterations - start_iterations < max_iterations)
                & (jnp.max(jnp.abs(s.residual)) > 1),
                newton_step,
                state,
            )
            status = jnp.where(
                (state.status == RceStatus.RUNNING)
                & (jnp.max(jnp.abs(state.residual)) > 1),
                RceStatus.MAX_ITERATIONS,
                state.status,
            )
            updated_mask = jnp.where(
                state.mask,
                flux - state.column.net_flux[1:] >= -flux_tolerance,
                state.column.stability_excess > stability_atol,
            )
            same_mask = jnp.all(updated_mask == state.mask)
            status = jnp.where(
                (status == RceStatus.RUNNING) & same_mask, RceStatus.CONVERGED, status
            )
            status = jnp.where(
                (status == RceStatus.RUNNING)
                & (state.active_iterations == max_active_set_iterations),
                RceStatus.MAX_ACTIVE_SET_ITERATIONS,
                status,
            )
            next_mask = jnp.where(status == RceStatus.RUNNING, updated_mask, state.mask)
            state = state._replace(
                mask=next_mask,
                status=status,
                residual=_scaled_residual(
                    state.column, next_mask, flux, flux_tolerance, stability_atol
                ),
            )
            state = state._replace(
                status=jnp.where(
                    (state.status == RceStatus.RUNNING)
                    & ~jnp.all(jnp.isfinite(state.residual)),
                    RceStatus.NONFINITE_RESIDUAL,
                    state.status,
                )
            )
            return state, history, history_count + 1

        state, _, _ = jax.lax.while_loop(
            lambda carry: (carry[0].status == RceStatus.RUNNING)
            & (carry[0].active_iterations < max_active_set_iterations),
            active_body,
            (state, history, zero),
        )
        values = jnp.exp(state.log_t)
        domain_valid = domain(state.log_t, parameters)
        # Check the actual returned temperatures, independently of the Newton
        # residual carry (including exp/log rounding in gradient adapters).
        column = jax.lax.cond(
            domain_valid & valid_flux,
            lambda: evaluate_values(values, parameters),
            lambda: empty,
        )
        convective_flux = jnp.concatenate(
            (
                jnp.zeros(1, dtype=dtype),
                jnp.where(state.mask, flux - column.net_flux[1:], 0.0),
            )
        )
        flux_residual = column.net_flux + convective_flux - flux
        scaled_residual = _scaled_residual(
            column, state.mask, flux, flux_tolerance, stability_atol
        )
        physics_valid = domain_valid & (column_status(column) == RceStatus.RUNNING)
        consistent = (
            physics_valid
            & jnp.all(jnp.isfinite(scaled_residual))
            & jnp.all(jnp.abs(scaled_residual) <= 1)
            & jnp.all(jnp.abs(flux_residual) <= flux_tolerance)
            & jnp.all(column.stability_excess <= stability_atol)
            & jnp.all(convective_flux >= -flux_tolerance)
        )
        state = state._replace(
            status=jnp.where(
                (state.status == RceStatus.CONVERGED) & ~consistent,
                RceStatus.INCONSISTENT_RESIDUAL,
                state.status,
            )
        )
        return DeviceRceResult(
            values[:-1],
            values[-1],
            column.net_flux,
            convective_flux,
            state.mask,
            flux_residual,
            column.stability_excess,
            scaled_residual,
            state.iterations,
            state.active_iterations,
            state.backtracks,
            domain_valid,
            physics_valid,
            column.physics_status,
            state.status == RceStatus.CONVERGED,
            state.status,
        )

    return solve
