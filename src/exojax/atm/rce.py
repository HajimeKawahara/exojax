"""Steady, dry radiative-convective equilibrium on a fixed pressure grid.

Temperatures live at layer centers and at a black lower boundary. Radiation
is supplied as a JAX-compatible callback returning net upward bolometric flux
at every interface, including incident stellar radiation. The host-controlled
active-set Newton iteration is not itself differentiable.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RceResult:
    """Last accepted state and convergence diagnostics.

    Fluxes and ``flux_residual`` are in erg/s/cm2 and have shape ``(N+1,)``.
    Convective flux is zero at the top and on inactive connections.
    ``flux_residual`` is the total upward flux minus the internal flux.
    ``gradient_residual`` is d log(T)/d log(P) minus the dry adiabat, with
    shape ``(N,)``; it must be nonpositive on inactive connections and zero
    on active ones. ``scaled_residual`` contains the equations selected by
    the mask, divided by their respective tolerances.
    """

    temperature: np.ndarray
    bottom_temperature: float
    radiative_flux: np.ndarray
    convective_flux: np.ndarray
    convective_mask: np.ndarray
    flux_residual: np.ndarray
    gradient_residual: np.ndarray
    scaled_residual: np.ndarray
    iterations: int
    active_set_iterations: int
    domain_valid: bool
    converged: bool
    status: str


def reconstruct_boundary_temperature(
    pressure_bar, pressure_boundaries_bar, temperature, bottom_temperature
):
    """Interpolate log T in log P from layer centers to RT interfaces.

    Arrays run from top to bottom. Center pressures and temperatures have
    shape ``(N,)``; boundary pressures have shape ``(N+1,)``. The top
    boundary takes the first layer temperature (an isothermal upper half
    layer), and the bottom takes ``bottom_temperature``. This reconstruction
    specifies interface sources. Also pass the center source to the linear
    RT flux routine, splitting each layer at its pressure center. Including
    both source locations avoids alternating temperature modes in thin layers.
    The source is linear in optical depth in each half; opacity is evaluated
    at layer centers and held constant through the layer.
    """
    pressure_nodes = jnp.append(pressure_bar, pressure_boundaries_bar[-1])
    temperature_nodes = jnp.append(temperature, bottom_temperature)
    return jnp.exp(
        jnp.interp(
            jnp.log(pressure_boundaries_bar),
            jnp.log(pressure_nodes),
            jnp.log(temperature_nodes),
        )
    )


def rce_residual(
    log_temperature,
    pressure_bar,
    bottom_pressure_bar,
    internal_flux,
    radiative_flux,
    adiabatic_gradient,
    convective_mask,
    flux_scale=1.0,
    gradient_scale=1.0,
):
    """Evaluate the reduced fixed-mask RCE equations with JAX arrays.

    ``log_temperature`` holds N center values followed by the bottom value.
    ``radiative_flux(T, T_bottom)`` returns N+1 net upward interface fluxes
    in erg/s/cm2. ``convective_mask`` and the scalar or N-element dry
    ``adiabatic_gradient`` describe center-to-center connections followed
    by the last center-to-bottom connection. The top equation always imposes
    radiative energy balance. Other equations impose either energy balance
    or a neutral dry gradient. Scales are positive scalar normalizations.
    """
    temperature = jnp.exp(log_temperature)
    flux_error = (
        radiative_flux(temperature[:-1], temperature[-1]) - internal_flux
    ) / flux_scale
    dlog_pressure = jnp.diff(jnp.log(jnp.append(pressure_bar, bottom_pressure_bar)))
    gradient_error = (
        jnp.diff(log_temperature) / dlog_pressure - adiabatic_gradient
    ) / gradient_scale
    return jnp.concatenate(
        (flux_error[:1], jnp.where(convective_mask, gradient_error, flux_error[1:]))
    )


def solve_rce(
    pressure_bar,
    pressure_boundaries_bar,
    temperature_initial,
    bottom_temperature_initial,
    internal_flux,
    radiative_flux,
    adiabatic_gradient=2.0 / 7.0,
    *,
    convective_mask_initial=None,
    valid_state=None,
    flux_atol=1.0e-3,
    flux_rtol=1.0e-6,
    gradient_atol=1.0e-6,
    max_iterations=50,
    max_active_set_iterations=30,
    max_backtracks=25,
):
    """Solve efficient dry RCE using damped Newton steps and an active set.

    Args:
        pressure_bar: Positive, increasing layer-center pressures, shape (N,).
        pressure_boundaries_bar: Increasing interface pressures, shape (N+1,).
            Each center must lie strictly inside its layer.
        temperature_initial: Positive initial layer temperatures in K, shape (N,).
        bottom_temperature_initial: Initial black-boundary temperature in K.
        internal_flux: Prescribed nonnegative net internal flux in erg/s/cm2.
        radiative_flux: JAX-compatible callable ``(T, T_bottom) -> F_net``.
            It must recompute temperature-dependent opacities and sources at
            each call and return N+1 interface fluxes, positive upward, with
            any downward stellar flux subtracted. The black bottom source
            uses T_bottom, not the internal effective temperature.
        adiabatic_gradient: Fixed dry d log(T)/d log(P), scalar or shape (N,).
        convective_mask_initial: Optional boolean array of N connections.
            The default starts with all connections radiative.
        valid_state: Optional host callable ``(T, T_bottom) -> bool``. Checked
            before radiation at every trial state. Use it to enforce opacity
            table domains, including their fixed pressure range. An invalid
            initial state raises ValueError; invalid Newton trials are shortened.
        flux_atol: Absolute energy-balance tolerance in erg/s/cm2.
        flux_rtol: Relative tolerance against abs(internal_flux), never against
            incident stellar flux. The flux tolerance is atol + rtol * abs(F_int).
        gradient_atol: Absolute tolerance for dry gradient neutrality/stability.
        max_iterations: Maximum Newton steps per active set.
        max_active_set_iterations: Maximum number of masks to solve.
        max_backtracks: Maximum line-search trials per Newton step.

    Returns:
        RceResult: Convergence requires the selected equations, subadiabatic
        inactive connections, and nonnegative active convective flux to pass.
        Failures return the last accepted state with ``converged=False`` and
        an explicit status. Iterations are numerical, not physical time steps.

    Notes:
        Enable JAX x64 at the call site for precision-sensitive columns. The
        iteration driver uses NumPy, while the residual and its Jacobian use
        JAX. This baseline assumes fixed composition and a prescribed dry
        adiabat; it does not model latent heat, scattering, or interior cooling.
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
        or np.any(boundaries <= 0.0)
        or np.any(np.diff(boundaries) <= 0.0)
        or np.any(pressure <= boundaries[:-1])
        or np.any(pressure >= boundaries[1:])
    ):
        raise ValueError("Pressures must increase, with every center inside its layer.")
    if np.ndim(bottom_temperature_initial) != 0:
        raise ValueError("bottom_temperature_initial must be scalar.")
    temperatures = np.append(initial, bottom_temperature_initial)
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("Initial temperatures must be finite and positive.")
    if (
        np.ndim(internal_flux) != 0
        or not np.isfinite(internal_flux)
        or internal_flux < 0
    ):
        raise ValueError("internal_flux must be a finite, nonnegative scalar.")
    adiabat = np.asarray(adiabatic_gradient, dtype=float)
    if adiabat.shape not in ((), (nlayer,)):
        raise ValueError("adiabatic_gradient must be scalar or have shape (N,).")
    adiabat = np.broadcast_to(adiabat, (nlayer,))
    if not np.all(np.isfinite(adiabat)) or np.any(adiabat <= 0.0):
        raise ValueError("adiabatic_gradient must be finite and positive.")
    for name, value, allow_zero in (
        ("flux_atol", flux_atol, True),
        ("flux_rtol", flux_rtol, True),
        ("gradient_atol", gradient_atol, False),
    ):
        if (
            np.ndim(value) != 0
            or not np.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(f"Invalid {name}.")
    flux_tolerance = flux_atol + flux_rtol * abs(internal_flux)
    if not np.isfinite(flux_tolerance) or flux_tolerance <= 0:
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
    if convective_mask_initial is None:
        mask = np.zeros(nlayer, dtype=bool)
    else:
        mask = np.asarray(convective_mask_initial)
        if mask.shape != (nlayer,) or mask.dtype != np.dtype(bool):
            raise ValueError(
                "convective_mask_initial must be a boolean array of shape (N,)."
            )
        mask = mask.copy()

    def in_domain(values):
        return bool(
            np.all(np.isfinite(values))
            and np.all(values > 0)
            and (valid_state is None or valid_state(values[:-1], values[-1]))
        )

    if not in_domain(temperatures):
        raise ValueError("Initial temperatures are outside the valid_state domain.")

    def residual(log_t, active):
        return rce_residual(
            log_t,
            jnp.asarray(pressure),
            boundaries[-1],
            internal_flux,
            radiative_flux,
            jnp.asarray(adiabat),
            active,
            flux_tolerance,
            gradient_atol,
        )

    @jax.jit
    def evaluate(log_t, active):
        values = jnp.exp(log_t)
        return residual(log_t, active), radiative_flux(values[:-1], values[-1])

    jacobian = jax.jit(jax.jacfwd(residual, argnums=0))
    log_t = np.log(temperatures)

    def state_temperature(log_values):
        # Use the same device precision and exponential as the RT callback.
        return np.asarray(jnp.exp(jnp.asarray(log_values)))

    initial_state = state_temperature(log_t)
    if not in_domain(initial_state):
        raise ValueError(
            "Initial temperatures are outside the valid_state domain in JAX precision."
        )
    initial_flux = np.asarray(
        radiative_flux(jnp.asarray(initial_state[:-1]), initial_state[-1])
    )
    if initial_flux.shape != (nlayer + 1,):
        raise ValueError("radiative_flux must return an array of shape (N+1,).")
    dlog_pressure = np.diff(np.log(np.append(pressure, boundaries[-1])))
    iterations = 0
    active_iterations = 0

    def result(status):
        values = state_temperature(log_t)
        flux = np.asarray(radiative_flux(jnp.asarray(values[:-1]), values[-1]))
        convective = np.concatenate(
            ([0.0], np.where(mask, internal_flux - flux[1:], 0.0))
        )
        with np.errstate(invalid="ignore"):
            flux_error = flux + convective - internal_flux
        gradient_error = (
            np.diff(np.log(values.astype(float))) / dlog_pressure - adiabat
        )
        # Independently check the returned physical state. JIT fusion and
        # host/device rounding can differ, especially without caller-enabled x64.
        if status == "converged" and not (
            np.all(np.isfinite(flux))
            and np.all(np.abs(flux_error) <= flux_tolerance)
            and np.all(np.abs(gradient_error[mask]) <= gradient_atol)
            and np.all(gradient_error[~mask] <= gradient_atol)
            and np.all(convective >= -flux_tolerance)
            and in_domain(values)
        ):
            status = "inconsistent_residual"
        return RceResult(
            temperature=values[:-1],
            bottom_temperature=float(values[-1]),
            radiative_flux=flux,
            convective_flux=convective,
            convective_mask=mask.copy(),
            flux_residual=flux_error,
            gradient_residual=gradient_error,
            scaled_residual=np.asarray(evaluate(log_t, mask)[0]),
            iterations=iterations,
            active_set_iterations=active_iterations,
            domain_valid=in_domain(values),
            converged=status == "converged",
            status=status,
        )

    seen_masks = set()
    for active_iterations in range(1, max_active_set_iterations + 1):
        key = mask.tobytes()
        if key in seen_masks:
            return result("active_set_cycle")
        seen_masks.add(key)
        for step_index in range(max_iterations + 1):
            error, full_flux = map(np.asarray, evaluate(log_t, mask))
            if not np.all(np.isfinite(error)) or not np.all(np.isfinite(full_flux)):
                return result("nonfinite_residual")
            norm = np.max(np.abs(error))
            if norm <= 1.0:
                break
            if step_index == max_iterations:
                return result("max_iterations")
            matrix = np.asarray(jacobian(log_t, mask))
            if not np.all(np.isfinite(matrix)):
                return result("nonfinite_jacobian")
            try:
                step = np.linalg.solve(matrix, -error)
            except np.linalg.LinAlgError:
                return result("singular_jacobian")
            if not np.all(np.isfinite(step)):
                return result("nonfinite_step")
            # A log-temperature step of one limits the largest change to e.
            damping = min(1.0, 1.0 / max(1.0, np.max(np.abs(step))))
            had_valid_trial = False
            accepted = False
            for _ in range(max_backtracks):
                trial = log_t + damping * step
                trial_temperature = state_temperature(trial)
                if in_domain(trial_temperature):
                    had_valid_trial = True
                    trial_error, trial_flux = map(np.asarray, evaluate(trial, mask))
                    if np.all(np.isfinite(trial_error)) and np.all(
                        np.isfinite(trial_flux)
                    ):
                        trial_norm = np.max(np.abs(trial_error))
                        if (
                            trial_norm <= 1.0
                            or trial_norm < (1.0 - 1.0e-4 * damping) * norm
                        ):
                            log_t = trial
                            accepted = True
                            break
                damping *= 0.5
            if not accepted:
                return result(
                    "line_search_failed" if had_valid_trial else "domain_step_failed"
                )
            iterations += 1

        values = state_temperature(log_t)
        flux = np.asarray(radiative_flux(jnp.asarray(values[:-1]), values[-1]))
        gradient_excess = (
            np.diff(np.log(values.astype(float))) / dlog_pressure - adiabat
        )
        updated = np.where(
            mask,
            internal_flux - flux[1:] >= -flux_tolerance,
            gradient_excess > gradient_atol,
        )
        if np.array_equal(updated, mask):
            return result("converged")
        if active_iterations < max_active_set_iterations:
            mask = updated
    return result("max_active_set_iterations")
