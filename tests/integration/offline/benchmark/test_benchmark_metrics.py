"""Analytic checks for benchmark observation and derivative decisions."""

import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmark"))
    return importlib.import_module("benchmark_metrics")


def test_observation_noise_scaling_and_both_error_budgets(metrics):
    result = metrics.observation_error([1.006, 2.016], [1.0, 2.0], [1.0, 2.0])
    assert result["passed"]
    assert result["max_error_in_noise"] == pytest.approx(0.008)
    assert result["q"] == pytest.approx(0.0001)
    assert result["maximum_error_index"] == [1]
    json.dumps(result, allow_nan=False)
    assert not metrics.observation_error([0.02], [0.0], 1.0)["passed"]
    result = metrics.observation_error(np.full(1100, 0.01), np.zeros(1100), 1.0)
    assert not result["passed"] and result["q"] > 0.1
    assert metrics.observation_error(1.0, 1.0, 0.1)["maximum_error_index"] == []


@pytest.mark.parametrize(
    "candidate, reference, noise",
    [
        ([], [], 1.0),
        ([1.0, 2.0], [1.0], 1.0),
        ([1.0, 2.0], [1.0, 2.0], [1.0]),
        ([1.0], [1.0], 0.0),
        ([1.0], [1.0], -1.0),
        ([1.0], [1.0], np.inf),
    ],
)
def test_observation_rejects_ambiguous_or_invalid_inputs(
    metrics, candidate, reference, noise
):
    with pytest.raises(ValueError):
        metrics.observation_error(candidate, reference, noise)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e308])
def test_nonfinite_observation_or_overflow_is_saved_as_failed(metrics, value):
    result = metrics.observation_error([0.0, value], [0.0, 0.0], 0.01)
    assert not result["finite"] and not result["passed"]
    assert result["max_error_in_noise"] is None and result["q"] is None
    assert result["maximum_error_index"] is None
    if not np.isfinite(value):
        assert result["nonfinite_indices"]["candidate"] == [[1]]
    json.dumps(result, allow_nan=False)


def test_q_overflow_is_reported_even_when_residuals_are_finite(metrics):
    result = metrics.observation_error([1e200], [0.0], 1.0)
    assert not result["finite"] and result["nonfinite_q"]
    assert result["nonfinite_indices"]["residual"] == []
    assert result["nonfinite_indices"]["squared_residual"] == [[0]]
    json.dumps(result, allow_nan=False)


def test_vector_and_scalar_directional_derivatives_have_known_values(metrics):
    def forward(x):
        return jnp.array([x[0] ** 2 + 3 * x[1], jnp.sin(x[1])])

    result = metrics.directional_check(forward, [2.0, 0.0], [0.25, 0.5])
    assert result["passed"]
    assert result["ad_directional_derivative"] == pytest.approx([2.5, 0.5])
    assert all(record["passed"] for record in result["steps"])
    result = metrics.directional_check(lambda x: x**3, 2.0, 0.5)
    assert result["passed"] and result["ad_directional_derivative"] == pytest.approx(
        6.0
    )
    json.dumps(result, allow_nan=False)


def test_dimensionless_scaling_keeps_absolute_error_near_zero(metrics):
    small = metrics.derivative_difference([1e-8, 0.0], [2e-8, 0.0])
    assert small["max_absolute"] == small["scaled"] == pytest.approx(1e-8)
    large = metrics.derivative_difference([100.0, 1.0], [99.0, 1.0])
    assert large["max_absolute"] == 1.0 and large["scaled"] == 0.01
    failed = metrics.derivative_difference([np.inf], [1.0])
    assert not failed["finite"] and failed["scaled"] is None
    json.dumps(failed, allow_nan=False)


def test_broken_custom_gradient_is_detected(metrics):
    @jax.custom_jvp
    def broken(x):
        return x**2

    @broken.defjvp
    def wrong_jvp(primals, tangents):
        (x,), (tangent,) = primals, tangents
        return broken(x), 3 * x * tangent

    result = metrics.directional_check(broken, 2.0, 1.0)
    assert not result["passed"]
    assert result["ad_directional_derivative"] == 6.0
    assert all(
        record["absolute_error"] == pytest.approx(2.0) for record in result["steps"]
    )


def test_excluded_steps_cannot_bridge_passing_steps(metrics):
    result = metrics.directional_check(
        lambda x: x**2,
        0.0,
        1.0,
        steps=[0.1, 0.01, 0.001],
        in_domain=lambda x: not np.isclose(abs(x), 0.01),
    )
    assert [record["passed"] for record in result["steps"]] == [True, False, True]
    assert result["steps"][1]["status"] == "out_of_domain"
    assert not result["passed"]


def test_clipping_kink_reports_both_sides_without_smooth_pass(metrics):
    result = metrics.directional_check(
        lambda x: jnp.clip(x, 0.0, 1.0),
        0.0,
        1.0,
        region=lambda x: np.sign(x),
    )
    assert not result["passed"]
    for record in result["steps"]:
        assert record["status"] == "kink" and not record["passed"]
        assert record["one_sided_derivatives"] == {"backward": 0.0, "forward": 1.0}
    json.dumps(result, allow_nan=False)


def test_domain_rejection_never_evaluates_invalid_points(metrics):
    def forward(x):
        if isinstance(x, np.ndarray):
            assert x >= 0
        return x**2

    result = metrics.directional_check(forward, 0.0, 1.0, in_domain=lambda x: x >= 0)
    assert not result["passed"]
    assert all(record["status"] == "out_of_domain" for record in result["steps"])
    result = metrics.directional_check(forward, -1.0, 1.0, in_domain=lambda x: x >= 0)
    assert result["reason"] == "center_out_of_domain"


def test_nonfinite_ad_is_not_hidden_by_finite_forward(metrics):
    @jax.custom_jvp
    def broken(x):
        return x

    @broken.defjvp
    def nonfinite_jvp(primals, tangents):
        return broken(primals[0]), jnp.full_like(tangents[0], jnp.nan)

    result = metrics.directional_check(broken, [1.0], [1.0])
    assert not result["passed"] and result["reason"] == "nonfinite_evaluation"
    assert result["ad_directional_derivative"] == [None]
    assert result["nonfinite_ad_indices"] == [[0]]
    json.dumps(result, allow_nan=False)


def test_a_nonfinite_difference_fails_even_with_other_passing_steps(metrics):
    result = metrics.directional_check(
        lambda x: jnp.where(jnp.abs(x) > 0.05, jnp.nan, x**2),
        0.0,
        1.0,
        steps=[0.1, 0.01, 0.001],
    )
    assert not result["passed"] and result["reason"] == "nonfinite_evaluation"
    assert result["steps"][0]["status"] == "nonfinite"
    assert all(record["passed"] for record in result["steps"][1:])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "steps", [[], [0.0], [-0.1], [np.nan], [0.01, 0.1], [0.1, 0.1]]
)
def test_invalid_step_sequences_are_rejected(metrics, steps):
    with pytest.raises(ValueError, match="Steps"):
        metrics.directional_check(lambda x: x**2, 1.0, 1.0, steps=steps)


def test_a_single_good_step_does_not_establish_convergence(metrics):
    result = metrics.directional_check(lambda x: x**2, 1.0, 1.0, steps=[0.01])
    assert result["steps"][0]["passed"] and not result["passed"]
