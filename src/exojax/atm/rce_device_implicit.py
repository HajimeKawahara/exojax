"""First-order implicit sensitivities and guarded device RCE likelihoods."""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
from jax.flatten_util import ravel_pytree

from exojax.atm.rce_device import (
    ColumnEvaluation,
    DeviceRceResult,
    _dense_jacobian,
    _scaled_residual,
    make_device_rce_solver,
)


class DerivativeStatus(IntEnum):
    """Why a local fixed-mask sensitivity was accepted or rejected."""

    VALID = 0
    PRIMAL_FAILED = 1
    CONVECTIVE_SWITCH = 2
    NONSMOOTH_PHYSICS = 3
    NONFINITE_JACOBIAN = 4
    SINGULAR_JACOBIAN = 5
    NONFINITE_SENSITIVITY = 6
    INACCURATE_LINEAR_SOLVE = 7


class DeviceImplicitRceResult(NamedTuple):
    """Primal state and an independent certificate for first-order AD.

    Continuous physical fields in ``state`` have implicit derivatives; status,
    counters, masks, and all fields outside ``state`` have zero tangents.
    ``complementarity_margin`` is convective flux / flux tolerance on active
    connections, and -stability excess / stability tolerance on inactive ones.
    It must exceed ``switch_margin``. ``linear_residual`` is the maximum error
    in A D + B, divided by the configured per-column linear solve tolerance;
    a passing certificate is <= 1. It is infinity if construction was skipped.
    """

    state: DeviceRceResult
    derivative_valid: jax.Array
    derivative_status: jax.Array
    complementarity_margin: jax.Array
    linear_residual: jax.Array


class LogProbStatus(IntEnum):
    """Prior support is distinct from numerical or differentiability failure."""

    ACCEPTED = 0
    PRIOR_EXCLUDED = 1
    PRIMAL_FAILED = 2
    DERIVATIVE_INVALID = 3
    LIKELIHOOD_EXCLUDED = 4
    NONFINITE_LOG_PROB = 5


class RceLogProbDiagnostics(NamedTuple):
    """Integer statuses; -1 denotes a stage skipped by an earlier guard."""

    status: jax.Array
    primal_status: jax.Array
    derivative_status: jax.Array


_PHYSICAL_FIELDS = (
    "temperature",
    "bottom_temperature",
    "radiative_flux",
    "convective_flux",
    "flux_residual",
    "stability_residual",
    "scaled_residual",
)


def _zero_tangent(tree):
    def zero(value):
        dtype = (
            value.dtype
            if jnp.issubdtype(value.dtype, jnp.inexact)
            else jax.dtypes.float0
        )
        return jnp.zeros(value.shape, dtype=dtype)

    return jax.tree_util.tree_map(zero, tree)


def _boolean_scalar(value, name):
    value = jnp.asarray(value)
    if value.shape != () or value.dtype != jnp.bool_:
        raise ValueError(f"{name} must return a scalar boolean.")
    return value


def make_device_implicit_rce_solver(
    pressure_bar,
    pressure_boundaries_bar,
    temperature_initial,
    bottom_temperature_initial,
    internal_flux,
    evaluate_column,
    *,
    local_smoothness=None,
    switch_margin=1.0,
    linear_atol=1.0e-10,
    linear_rtol=1.0e-8,
    invalid_derivative="nan",
    **solver_options,
):
    """Prepare ``solve(parameters)`` with device-only, fixed-mask implicit AD.

    The primal arguments/options are those of ``make_device_rce_solver``.
    Parameters must be a nonempty PyTree of real floating-point leaves. The
    selected residual and Jacobian construction mode are shared with that
    solver. All dependencies must be explicit in parameters. Only first-order
    JVP/VJP are supported; the active-set/Newton iteration is not differentiated.

    Before constructing local Jacobians, require primal convergence, strict
    convective inequalities, and optional ``local_smoothness(T, T_bottom, p)``.
    The caller must supply this predicate at interpolation knots, domain
    boundaries, or other nonsmooth physics; smoothness cannot be inferred from
    finite derivatives. Both temperature and parameter Jacobians must be
    finite. Solve A D = -B without regularization and check its residual for
    every parameter direction. This uses dense matrices including one column
    per scalar parameter, intended for small retrieval parameter vectors.

    ``invalid_derivative='nan'`` gives strict diagnostic NaN sensitivities.
    ``'zero'`` supplies finite zero tangents solely for guarded rejection, while
    retaining ``derivative_valid=False`` and the failure status. These zeros
    are not physical sensitivities. Use the latter with ``make_rce_log_prob``;
    an outer ``where`` cannot safely hide NaNs from the strict path.
    For mixed valid/invalid batches use ``lax.map``, not unrestricted ``vmap``.
    """
    for name, value in (
        ("switch_margin", switch_margin),
        ("linear_atol", linear_atol),
        ("linear_rtol", linear_rtol),
    ):
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if switch_margin < 1:
        raise ValueError("switch_margin must be at least one primal tolerance.")
    if linear_atol == 0 and linear_rtol == 0:
        raise ValueError("The linear solve tolerance must be positive.")
    if invalid_derivative not in ("nan", "zero"):
        raise ValueError("invalid_derivative must be 'nan' or 'zero'.")
    if local_smoothness is not None and not callable(local_smoothness):
        raise TypeError("local_smoothness must be callable.")
    dtype = jnp.asarray(temperature_initial, dtype=float).dtype
    for name, value in (("linear_atol", linear_atol), ("linear_rtol", linear_rtol)):
        if value > 0 and (value < np.finfo(dtype).tiny or value > np.finfo(dtype).max):
            raise ValueError(
                f"{name} must be representable as a normal value in {dtype}."
            )
    switch_margin, linear_atol, linear_rtol = map(
        float, (switch_margin, linear_atol, linear_rtol)
    )

    primal_solve = make_device_rce_solver(
        pressure_bar,
        pressure_boundaries_bar,
        temperature_initial,
        bottom_temperature_initial,
        internal_flux,
        evaluate_column,
        **solver_options,
    )
    flux_atol = float(solver_options.get("flux_atol", 1.0e-3))
    flux_rtol = float(solver_options.get("flux_rtol", 1.0e-6))
    stability_atol = float(solver_options.get("stability_atol", 1.0e-6))
    mode = solver_options.get("jacobian_mode", "sequential")

    def flux(parameters):
        return internal_flux(parameters) if callable(internal_flux) else internal_flux

    def solve_and_linearize(parameters):
        # The primal is executed once and is never differentiated through.
        state = jax.tree_util.tree_map(jax.lax.stop_gradient, primal_solve(parameters))
        flat_parameters, unpack_parameters = ravel_pytree(parameters)
        physical, unpack_physical = ravel_pytree(
            tuple(getattr(state, name) for name in _PHYSICAL_FIELDS)
        )
        log_t = jnp.log(jnp.append(state.temperature, state.bottom_temperature))
        tolerance = flux_atol + flux_rtol * jnp.abs(flux(parameters))
        margin = jnp.where(
            state.convective_mask,
            state.convective_flux[1:] / tolerance,
            -state.stability_residual / stability_atol,
        )
        primal_valid = state.converged & state.domain_valid & state.physics_valid
        strict = jnp.all(jnp.isfinite(margin) & (margin > switch_margin))

        def smooth():
            if local_smoothness is None:
                return jnp.asarray(True)
            return _boolean_scalar(
                local_smoothness(
                    state.temperature, state.bottom_temperature, parameters
                ),
                "local_smoothness",
            )

        smooth_valid = jax.lax.cond(
            primal_valid & strict, smooth, lambda: jnp.asarray(False)
        )
        status = jnp.where(
            ~primal_valid,
            DerivativeStatus.PRIMAL_FAILED,
            jnp.where(
                ~strict,
                DerivativeStatus.CONVECTIVE_SWITCH,
                jnp.where(
                    smooth_valid,
                    DerivativeStatus.VALID,
                    DerivativeStatus.NONSMOOTH_PHYSICS,
                ),
            ),
        ).astype(jnp.int32)

        def outputs(u, flat_p):
            p = unpack_parameters(flat_p)
            values = jnp.exp(u)
            raw = evaluate_column(values[:-1], values[-1], p)
            # Match the primal evaluator's precision before residual scaling.
            column = ColumnEvaluation(
                jnp.asarray(raw.net_flux, dtype=u.dtype),
                jnp.asarray(raw.stability_excess, dtype=u.dtype),
                raw.physics_valid,
                raw.physics_status,
            )
            internal = jnp.asarray(flux(p), dtype=u.dtype)
            flux_tolerance = flux_atol + flux_rtol * jnp.abs(internal)
            residual = _scaled_residual(
                column, state.convective_mask, internal, flux_tolerance, stability_atol
            )
            convective = jnp.concatenate(
                (
                    jnp.zeros(1, dtype=u.dtype),
                    jnp.where(
                        state.convective_mask, internal - column.net_flux[1:], 0.0
                    ),
                )
            )
            fields = (
                values[:-1],
                values[-1],
                column.net_flux,
                convective,
                column.net_flux + convective - internal,
                column.stability_excess,
                residual,
            )
            packed, _ = ravel_pytree(fields)
            return jnp.concatenate((residual, packed))

        empty_response = jnp.zeros(
            (physical.size, flat_parameters.size), dtype=physical.dtype
        )

        def construct():
            jac_u = _dense_jacobian(lambda u: outputs(u, flat_parameters), log_t, mode)
            jac_p = _dense_jacobian(lambda p: outputs(log_t, p), flat_parameters, mode)
            matrix, rhs = jac_u[: log_t.size], jac_p[: log_t.size]
            finite = jnp.all(jnp.isfinite(jac_u)) & jnp.all(jnp.isfinite(jac_p))

            def solve_linear():
                lu, pivots = jlinalg.lu_factor(matrix)
                nonsingular = jnp.all(jnp.isfinite(lu)) & jnp.all(jnp.diag(lu) != 0)

                def nonsingular_solve():
                    response_u = jlinalg.lu_solve((lu, pivots), -rhs)
                    response = jac_u[log_t.size :] @ response_u + jac_p[log_t.size :]
                    limit = linear_atol + linear_rtol * jnp.max(jnp.abs(rhs), axis=0)
                    absolute_error = jnp.max(jnp.abs(matrix @ response_u + rhs), axis=0)
                    positive_limit = limit >= np.finfo(dtype).tiny
                    normalized = absolute_error / jnp.where(positive_limit, limit, 1.0)
                    normalized = jnp.where(
                        positive_limit,
                        normalized,
                        jnp.where(absolute_error == 0, 0.0, jnp.inf),
                    )
                    error = jnp.max(normalized)
                    finite_response = jnp.all(jnp.isfinite(response_u)) & jnp.all(
                        jnp.isfinite(response)
                    )
                    status = jnp.where(
                        ~finite_response,
                        DerivativeStatus.NONFINITE_SENSITIVITY,
                        jnp.where(
                            jnp.all(jnp.isfinite(limit))
                            & jnp.isfinite(error)
                            & (error <= 1),
                            DerivativeStatus.VALID,
                            DerivativeStatus.INACCURATE_LINEAR_SOLVE,
                        ),
                    ).astype(jnp.int32)
                    return response, status, error

                return jax.lax.cond(
                    nonsingular,
                    nonsingular_solve,
                    lambda: (
                        empty_response,
                        jnp.int32(DerivativeStatus.SINGULAR_JACOBIAN),
                        jnp.asarray(jnp.inf, physical.dtype),
                    ),
                )

            return jax.lax.cond(
                finite,
                solve_linear,
                lambda: (
                    empty_response,
                    jnp.int32(DerivativeStatus.NONFINITE_JACOBIAN),
                    jnp.asarray(jnp.inf, physical.dtype),
                ),
            )

        response, status, error = jax.lax.cond(
            status == DerivativeStatus.VALID,
            construct,
            lambda: (empty_response, status, jnp.asarray(jnp.inf, physical.dtype)),
        )
        result = DeviceImplicitRceResult(
            state, status == DerivativeStatus.VALID, status, margin, error
        )
        return result, response, unpack_physical

    @jax.custom_jvp
    def equilibrium(parameters):
        return solve_and_linearize(parameters)[0]

    @equilibrium.defjvp
    def equilibrium_jvp(primals, tangents):
        (parameters,), (parameter_tangent,) = primals, tangents
        result, response, unpack_physical = solve_and_linearize(parameters)
        tangent, _ = ravel_pytree(parameter_tangent)

        def rejected_tangent():
            if invalid_derivative == "zero":
                return jnp.zeros(response.shape[0], response.dtype)
            # This rule stays linear in the input tangent, including in the
            # strict path, so reverse AD returns NaNs for invalid sensitivities.
            return jnp.full(response.shape[0], jnp.nan, response.dtype) * jnp.sum(
                tangent
            )

        physical_tangent = jax.lax.cond(
            result.derivative_valid, lambda: response @ tangent, rejected_tangent
        )
        result_tangent = _zero_tangent(result)
        state_tangent = result_tangent.state._replace(
            **dict(zip(_PHYSICAL_FIELDS, unpack_physical(physical_tangent)))
        )
        return result, result_tangent._replace(state=state_tangent)

    @jax.jit
    def solve(parameters):
        parameters = jax.tree_util.tree_map(jnp.asarray, parameters)
        leaves = jax.tree_util.tree_leaves(parameters)
        if not leaves or not sum(leaf.size for leaf in leaves):
            raise TypeError(
                "parameters must contain at least one floating-point value."
            )
        if any(not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves):
            raise TypeError("parameters must contain only real floating-point leaves.")
        return equilibrium(parameters)

    solve.invalid_derivative = invalid_derivative
    return solve


def make_rce_log_prob(
    solve, log_prob_from_state, *, prior_valid=None, likelihood_valid=None
):
    """Guard an RCE log density and return ``(value, RceLogProbDiagnostics)``.

    ``solve`` must use ``invalid_derivative='zero'``. The callback takes
    ``(DeviceRceResult, parameters)`` and includes any prior density. It must be
    a scalar floating-point value and locally smooth at allowed states.
    Optional scalar boolean guards ``prior_valid(parameters)`` and
    ``likelihood_valid(state, parameters)``
    define its domain: the former runs before solving, the latter before the
    log density. Solver/sensitivity failures also skip downstream physics.

    Nonfinite log density values are rejected before constructing their AD rule.
    Rejection returns -inf with finite zero gradients and explicit diagnostics;
    numerical failure is never relabeled as prior exclusion. Use
    ``jax.jit(jax.value_and_grad(log_prob, has_aux=True))``. A non-negligible
    failure rate inside the intended prior requires fixing the model/solver,
    since rejection otherwise changes the sampled distribution. This helper
    cannot detect arbitrary singular derivatives inside the likelihood itself.
    """
    if getattr(solve, "invalid_derivative", None) != "zero":
        raise ValueError("The guarded log density requires invalid_derivative='zero'.")
    if not callable(log_prob_from_state) or any(
        fn is not None and not callable(fn) for fn in (prior_valid, likelihood_valid)
    ):
        raise TypeError("The log density and its guards must be callable.")

    def density_value(state, parameters):
        value = jnp.asarray(log_prob_from_state(state, parameters))
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(
                "log_prob_from_state must return a scalar floating-point value."
            )
        return value.astype(jnp.asarray(0.0).dtype)

    @jax.custom_jvp
    def finite_density(state, parameters):
        value = density_value(state, parameters)
        valid = jnp.isfinite(value)
        return jnp.where(valid, value, -jnp.inf), valid

    @finite_density.defjvp
    def finite_density_jvp(primals, tangents):
        value = density_value(*primals)
        valid = jnp.isfinite(value)
        # A nonfinite primal must not enter likelihood differentiation. Merely
        # masking its final gradient could retain 0 * NaN in reverse mode.
        tangent = jax.lax.cond(
            valid,
            lambda: jax.jvp(density_value, primals, tangents)[1],
            lambda: jnp.zeros_like(value),
        )
        return (jnp.where(valid, value, -jnp.inf), valid), (
            tangent,
            jnp.zeros((), dtype=jax.dtypes.float0),
        )

    @jax.jit
    def log_prob(parameters):
        dtype = jnp.asarray(0.0).dtype
        rejected = jnp.asarray(-jnp.inf, dtype=dtype)
        prior_ok = (
            jnp.asarray(True)
            if prior_valid is None
            else _boolean_scalar(prior_valid(parameters), "prior_valid")
        )

        def within_prior():
            result = solve(parameters)

            def permitted():
                return (
                    jnp.asarray(True)
                    if likelihood_valid is None
                    else _boolean_scalar(
                        likelihood_valid(result.state, parameters), "likelihood_valid"
                    )
                )

            likelihood_ok = jax.lax.cond(
                result.derivative_valid, permitted, lambda: jnp.asarray(False)
            )

            value, finite = jax.lax.cond(
                likelihood_ok,
                lambda: finite_density(result.state, parameters),
                lambda: (rejected, jnp.asarray(False)),
            )
            status = jnp.where(
                ~result.state.converged,
                LogProbStatus.PRIMAL_FAILED,
                jnp.where(
                    ~result.derivative_valid,
                    LogProbStatus.DERIVATIVE_INVALID,
                    jnp.where(
                        ~likelihood_ok,
                        LogProbStatus.LIKELIHOOD_EXCLUDED,
                        jnp.where(
                            finite,
                            LogProbStatus.ACCEPTED,
                            LogProbStatus.NONFINITE_LOG_PROB,
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            return value, RceLogProbDiagnostics(
                status, result.state.status, result.derivative_status
            )

        return jax.lax.cond(
            prior_ok,
            within_prior,
            lambda: (
                rejected,
                RceLogProbDiagnostics(
                    jnp.int32(LogProbStatus.PRIOR_EXCLUDED),
                    jnp.int32(-1),
                    jnp.int32(-1),
                ),
            ),
        )

    return log_prob
