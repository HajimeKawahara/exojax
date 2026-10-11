"""Reachability and independent-point contracts of the primal continuation."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce_device import ColumnEvaluation, RceStatus


@pytest.fixture
def continuation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples"))
    return importlib.import_module("_rce_wasp18b_continuation").PreparedMaskContinuation


def _column():
    # The radiative subproblem wants Tb=5000 outside the domain, but a valid
    # convective root exists at Tb=3000 with positive transport of 40.
    def evaluate(t, tb, parameters):
        return ColumnEvaluation(
            jnp.stack([100 + t[0] - (2000 + parameters[0]),
                       100 + .02 * (tb - 5000)]),
            jnp.atleast_1d((tb - 3000) / 3000), jnp.array(True), jnp.int32(0),
        )

    def valid(t, tb, parameters):
        nodes = jnp.append(t, tb)
        return jnp.all((nodes >= 1500) & (nodes <= 4500)) & (parameters[0] >= 0)

    return SimpleNamespace(
        solver_arguments=([1.], [.1, 10.], [2000.], 2500., 100., evaluate),
        solver_options=dict(flux_atol=.1, flux_rtol=0., stability_atol=1e-8),
        temperature_valid=valid, internal_flux=100.,
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_continuation_reaches_exact_root_and_resets_each_point(continuation, explicit_none):
    column = _column()
    if explicit_none:
        column.solver_options["convective_mask_initial"] = None
    original_options = dict(column.solver_options)
    prepared = continuation(column, jnp.array([0.]))
    recorded = []
    first = prepared.run(jnp.array([0.]), on_stage=recorded.append)
    assert first.stop_reason == "converged" and first.state.converged
    assert len(first.stages) > 1 and len(recorded) == len(first.stages)
    assert not bool(first.stages[0]["state"]["converged"])
    assert first.stages[0]["next_convective_mask"] == [True]
    np.testing.assert_allclose(first.state.temperature, [2000.], atol=1e-8)
    np.testing.assert_allclose(first.state.bottom_temperature, 3000., atol=1e-6)
    np.testing.assert_allclose(first.state.convective_flux, [0., 40.], atol=1e-6)
    assert np.max(np.abs(first.state.flux_residual)) <= .1
    assert np.max(first.state.stability_residual) <= 1e-8

    other = prepared.run(jnp.array([20.]))
    repeated = prepared.run(jnp.array([0.]))
    for result in (first, other, repeated):
        assert result.state.converged
        assert result.stages[0]["initial_temperature_K"] == [2000.]
        assert result.stages[0]["initial_bottom_temperature_K"] == 2500.
        assert result.stages[0]["initial_convective_mask"] == [False]
    np.testing.assert_allclose(other.state.temperature, [2020.], atol=1e-8)
    for actual, expected in zip(first.state, repeated.state):
        np.testing.assert_array_equal(actual, expected)
    assert column.solver_options == original_options


def test_stage_budget_retains_failed_state_and_invalid_physics_stops(continuation):
    prepared = continuation(_column(), jnp.array([0.]), max_stages=1)
    result = prepared.run(jnp.array([0.]))
    assert result.stop_reason == "stage_budget_exhausted"
    assert not result.state.converged and len(result.stages) == 1
    assert result.state.status == RceStatus.MAX_ITERATIONS

    invalid = prepared.run(jnp.array([-1.]))
    assert invalid.stop_reason == "invalid_state" and len(invalid.stages) == 1
    assert invalid.state.status == RceStatus.INVALID_INITIAL_STATE
    assert not invalid.state.converged


def test_identical_failed_mask_cycle_stops_without_accepting_it(continuation):
    prepared = continuation(_column(), jnp.array([0.]))
    template = prepared.run(jnp.array([0.])).state
    calls = []

    def stalled(parameters, initial):
        temperature, bottom, mask = initial
        calls.append(np.asarray(mask).tolist())
        return template._replace(
            temperature=temperature, bottom_temperature=bottom, convective_mask=mask,
            radiative_flux=jnp.array([100., 101.]), stability_residual=jnp.array([1.]),
            converged=jnp.array(False), iterations=jnp.int32(0),
            status=jnp.int32(RceStatus.DOMAIN_STEP_FAILED))

    prepared.solve = stalled
    result = prepared.run(jnp.array([0.]))
    assert calls == [[False], [True]]
    assert result.stop_reason == "repeated_failed_state"
    assert not result.state.converged
    assert len(result.stages) == 2
    assert [stage["next_convective_mask"] for stage in result.stages] == [[True], [False]]


@pytest.mark.parametrize("options", [
    {"max_stages": 0}, {"max_stages": True}, {"newton_steps_per_stage": 0},
    {"newton_steps_per_stage": 1.5},
])
def test_iteration_budgets_are_positive_integers(continuation, options):
    with pytest.raises(ValueError, match="positive integer"):
        continuation(_column(), jnp.array([0.]), **options)
