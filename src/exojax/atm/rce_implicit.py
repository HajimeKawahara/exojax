"""Implicit derivatives of converged RCE states on a fixed pressure grid."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.rce import rce_residual, solve_rce


class ImplicitRceResult(NamedTuple):
    """JAX-compatible equilibrium state; failed solves have NaN temperatures.

    ``converged`` describes the primal solve, not the validity of a derivative.
    Sensitivities require a nonsingular residual Jacobian, strict convective
    inequalities, and locally smooth caller-supplied physics.
    """

    temperature: jax.Array
    bottom_temperature: jax.Array
    convective_mask: jax.Array
    converged: jax.Array


def make_implicit_rce_solver(
    pressure_bar,
    pressure_boundaries_bar,
    temperature_initial,
    bottom_temperature_initial,
    internal_flux,
    radiative_flux,
    neutral_gradient,
    *,
    valid_state=None,
    **solver_options,
):
    """Build a differentiable ``solve(parameters)`` using the host RCE solver.

    Construct this factory outside JAX transformations. Pressure arrays,
    initial guesses, and solver options are fixed. ``parameters`` is an
    explicit PyTree of real floating-point arrays or scalars. Radiation and
    neutral-gradient callbacks take ``(T, T_bottom, parameters)``; the gradient
    may instead be a constant scalar or array. ``internal_flux`` is a constant
    or a callable ``(parameters)``. An optional host ``valid_state`` callback
    takes ``(T, T_bottom, parameters)``. Remaining options go to ``solve_rce``.

    Every primal call solves the equilibrium for the supplied parameters.
    A custom JVP solves R_u du = -R_parameters dparameters at the converged
    log temperatures, holding the final convective mask fixed. JAX transposes
    this rule for reverse-mode differentiation; Newton iterations and initial
    guesses are not differentiated. Temperature-dependent radiation and
    neutral gradients participate in both residual derivatives.

    Expected solver failures, including ValueError from invalid states, return
    NaN temperatures and ``converged=False``. Use ``solve_rce`` for detailed
    failure diagnostics. At convective switches (within the solver tolerances)
    the derivative is NaN even if the primal converges. A singular Jacobian
    also yields nonfinite sensitivities; no regularization is applied. Other
    branch changes inside the physics callbacks remain the caller's concern.

    ``jax.jit``, ``jax.jvp``, and ``jax.grad`` are supported. The forward solve
    runs on the host through ``jax.pure_callback`` and needs a CPU backend;
    callbacks must be pure and must not explicitly pin work to a GPU. All
    differentiable dependencies must be passed in ``parameters``, not hidden
    in closures. Enable JAX x64 before constructing precision-sensitive solvers.
    """
    pressure = np.array(pressure_bar, dtype=float)
    boundaries = np.array(pressure_boundaries_bar, dtype=float)
    initial = np.array(temperature_initial, dtype=float)
    if pressure.ndim != 1 or pressure.size == 0:
        raise ValueError("pressure_bar must be a nonempty one-dimensional array.")
    nlayer = pressure.size
    if boundaries.shape != (nlayer + 1,) or initial.shape != (nlayer,):
        raise ValueError("Expected N center temperatures and N+1 pressure boundaries.")
    if np.ndim(bottom_temperature_initial) != 0:
        raise ValueError("bottom_temperature_initial must be scalar.")
    bottom_initial = float(bottom_temperature_initial)
    dtype = jnp.asarray(0.0).dtype
    flux_atol = solver_options.get("flux_atol", 1.0e-3)
    flux_rtol = solver_options.get("flux_rtol", 1.0e-6)
    gradient_atol = solver_options.get("gradient_atol", 1.0e-6)

    def flux(parameters):
        return internal_flux(parameters) if callable(internal_flux) else internal_flux

    def gradient(temperature, bottom, parameters):
        if callable(neutral_gradient):
            return neutral_gradient(temperature, bottom, parameters)
        return neutral_gradient

    def host_solve(parameters):
        # Materialize callback inputs before dispatching JAX work on the CPU.
        parameters = jax.tree_util.tree_map(np.array, parameters)
        packed = np.zeros(2 * nlayer + 2, dtype=dtype)
        packed[:nlayer + 1] = np.nan
        try:
            result = solve_rce(
                pressure, boundaries, initial, bottom_initial,
                np.asarray(flux(parameters)),
                lambda t, tb: radiative_flux(t, tb, parameters),
                lambda t, tb: gradient(t, tb, parameters),
                valid_state=(None if valid_state is None else
                             lambda t, tb: valid_state(t, tb, parameters)),
                **solver_options,
            )
        except ValueError:
            return packed
        packed[nlayer + 1:-1] = result.convective_mask
        packed[-1] = result.converged
        if result.converged:
            packed[:nlayer + 1] = np.log(
                np.append(result.temperature, result.bottom_temperature)
            )
        return packed

    @jax.custom_jvp
    def equilibrium(parameters):
        # Float auxiliaries avoid boolean/float0 custom-JVP issues on older JAX.
        return jax.pure_callback(
            host_solve, jax.ShapeDtypeStruct((2 * nlayer + 2,), dtype), parameters
        )

    @equilibrium.defjvp
    def equilibrium_jvp(primals, tangents):
        (parameters,), (parameter_tangent,) = primals, tangents
        packed = equilibrium(parameters)
        log_t = packed[:nlayer + 1]
        mask = packed[nlayer + 1:-1] > 0.5

        def residual(values, params):
            return rce_residual(
                values, jnp.asarray(pressure), boundaries[-1], flux(params),
                lambda t, tb: radiative_flux(t, tb, params),
                lambda t, tb: gradient(t, tb, params), mask,
            )

        matrix = jax.jacfwd(residual, argnums=0)(log_t, parameters)
        rhs = jax.jvp(
            lambda params: residual(log_t, params),
            (parameters,), (parameter_tangent,),
        )[1]
        # Row scaling preserves the equations while reducing unit imbalance.
        scale = jnp.max(jnp.abs(matrix), axis=1)
        scale = jnp.where(scale > 0.0, scale, 1.0)
        log_t_tangent = jnp.linalg.solve(matrix / scale[:, None], -rhs / scale)

        temperature = jnp.exp(log_t)
        rad = radiative_flux(temperature[:-1], temperature[-1], parameters)
        critical = gradient(temperature[:-1], temperature[-1], parameters)
        dlog_p = jnp.diff(jnp.log(jnp.append(pressure, boundaries[-1])))
        excess = jnp.diff(log_t) / dlog_p - critical
        flux_tolerance = flux_atol + flux_rtol * jnp.abs(flux(parameters))
        strict = jnp.all(jnp.where(
            mask, flux(parameters) - rad[1:] > flux_tolerance,
            excess < -gradient_atol,
        ))
        valid = (packed[-1] > 0.5) & strict
        # Keep the tangent rule linear, including when transposed by grad.
        log_t_tangent = jnp.where(valid, 1.0, jnp.nan) * log_t_tangent
        # Propagate invalidity even to parameters absent from the selected
        # equations (for example a flux parameter on an active connection).
        parameter_sum = sum(jnp.sum(leaf) for leaf in
                            jax.tree_util.tree_leaves(parameter_tangent))
        log_t_tangent += jnp.where(valid, 0.0, jnp.nan) * parameter_sum
        return packed, jnp.concatenate((
            log_t_tangent, jnp.zeros(nlayer + 1, dtype=dtype),
        ))

    def solve(parameters):
        parameters = jax.tree_util.tree_map(jnp.asarray, parameters)
        if any(not jnp.issubdtype(leaf.dtype, jnp.floating)
               for leaf in jax.tree_util.tree_leaves(parameters)):
            raise TypeError("parameters must contain only real floating-point leaves.")
        packed = equilibrium(parameters)
        temperature = jnp.exp(packed[:nlayer + 1])
        return ImplicitRceResult(
            temperature[:-1], temperature[-1],
            packed[nlayer + 1:-1] > 0.5, packed[-1] > 0.5,
        )

    return solve
