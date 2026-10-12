"""Bounded, within-point active-mask continuation for the WASP-18b primal.

Each call starts from the same explicit column seed. Failed fixed-mask stages
may update the mask using the production solver's complementarity test. Only
the production solver's exact CONVERGED result is accepted. This host caller
does not provide implicit derivatives or certify the opacity model/prior.
"""

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.rce_device import (
    DeviceRceResult, RceStatus, make_device_rce_solver,
)


@dataclass(frozen=True)
class ContinuationResult:
    state: DeviceRceResult
    stages: tuple
    stop_reason: str


class PreparedMaskContinuation:
    """Compile once, then solve points independently from the column's seed.

    ``run(parameters, on_stage=None)`` records every returned state, including
    failures. The optional callback receives each stage before the next solve,
    allowing the caller to checkpoint progress. Temperatures and masks are
    carried only between stages of that call, never between parameter points.
    """

    def __init__(self, column, parameters, *, newton_steps_per_stage=10,
                 max_stages=15):
        for name, value in (("newton_steps_per_stage", newton_steps_per_stage),
                            ("max_stages", max_stages)):
            if (not isinstance(value, (int, np.integer))
                    or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
        self.column = column
        self.max_stages = int(max_stages)
        self.solver_options = dict(column.solver_options)
        self.solver_options.update(max_iterations=int(newton_steps_per_stage),
                                   max_active_set_iterations=1)
        arguments = column.solver_arguments
        initial_mask = self.solver_options.get("convective_mask_initial")
        if initial_mask is None:
            initial_mask = np.zeros(len(arguments[2]), dtype=bool)
        self.initial_state = (
            jnp.array(arguments[2], dtype=float, copy=True),
            jnp.array(arguments[3], dtype=float, copy=True),
            jnp.array(initial_mask, copy=True),
        )
        solve = make_device_rce_solver(
            *arguments, valid_temperature=column.temperature_valid,
            **self.solver_options,
        )
        started = perf_counter()
        self.solve = solve.lower(jnp.asarray(parameters, dtype=float),
                                 self.initial_state).compile()
        self.compile_seconds = perf_counter() - started

    def run(self, parameters, on_stage=None):
        parameters = jnp.asarray(parameters, dtype=float)
        initial_state = self.initial_state
        stages = []
        previous_inputs = []
        flux = float(self.column.internal_flux)
        tolerance = (self.solver_options.get("flux_atol", 1e-3)
                     + self.solver_options.get("flux_rtol", 1e-6) * abs(flux))
        stability_atol = self.solver_options.get("stability_atol", 1e-6)
        recoverable = (RceStatus.MAX_ITERATIONS, RceStatus.MAX_ACTIVE_SET_ITERATIONS,
                       RceStatus.LINE_SEARCH_FAILED, RceStatus.DOMAIN_STEP_FAILED)
        stop_reason = "stage_budget_exhausted"
        for index in range(self.max_stages):
            previous_inputs.append(tuple(np.asarray(value).copy() for value in initial_state))
            started = perf_counter()
            state = jax.block_until_ready(self.solve(parameters, initial_state))
            mask = np.asarray(state.convective_mask)
            next_mask = np.where(
                mask, flux - np.asarray(state.radiative_flux)[1:] >= -tolerance,
                np.asarray(state.stability_residual) > stability_atol,
            )
            stage = {
                "index": index, "seconds": perf_counter() - started,
                "initial_temperature_K": np.asarray(initial_state[0]).tolist(),
                "initial_bottom_temperature_K": float(initial_state[1]),
                "initial_convective_mask": np.asarray(initial_state[2]).tolist(),
                "next_convective_mask": next_mask.tolist(),
                "status_name": RceStatus(int(state.status)).name,
                "state": state._asdict(),
            }
            stages.append(stage)
            if on_stage is not None:
                on_stage(stage)
            if bool(state.converged) and int(state.status) == RceStatus.CONVERGED:
                stop_reason = "converged"
                break
            if not bool(state.domain_valid) or not bool(state.physics_valid):
                stop_reason = "invalid_state"
                break
            if int(state.status) not in recoverable:
                stop_reason = "solver_failure"
                break
            if int(state.iterations) == 0 and np.array_equal(next_mask, initial_state[2]):
                stop_reason = "unchanged_failed_state"
                break
            initial_state = (state.temperature, state.bottom_temperature,
                             jnp.asarray(next_mask))
            if any(all(np.array_equal(value, previous)
                       for value, previous in zip(initial_state, earlier))
                   for earlier in previous_inputs):
                # Identical inputs to the same compiled deterministic solve
                # cannot escape a failed mask cycle on a later repetition.
                stop_reason = "repeated_failed_state"
                break
        return ContinuationResult(state, tuple(stages), stop_reason)
