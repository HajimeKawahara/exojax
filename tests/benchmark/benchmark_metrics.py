"""Small observation-space and directional-derivative validation helpers.

These are internal benchmark metrics. They do not select JAX precision or a
device; the caller supplies a forward model whitened by observational noise
and parameter coordinates scaled by their prior widths.
"""

import numpy as np


def _array(value):
    return np.asarray(value, dtype=float)


def _json_array(value):
    array = _array(value)
    return np.where(np.isfinite(array), array, None).tolist()


def _nonfinite(value):
    return np.argwhere(~np.isfinite(value)).tolist()


def _threshold(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


def derivative_difference(left, right):
    """Compare vectors/scalars without assigning a scientific pass threshold.

    Scaling is max(abs(left - right)) / max(1, max(abs(left)), max(abs(right))).
    The unit floor handles derivatives near zero and assumes a whitened forward
    model (or a dimensionless log density) in dimensionless parameter coordinates.
    """
    left, right = _array(left), _array(right)
    if left.shape != right.shape or not left.size:
        raise ValueError("Derivative arrays must have matching nonempty shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
    finite = bool(
        np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.all(np.isfinite(difference))
    )
    absolute = float(np.max(np.abs(difference))) if finite else None
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return {
        "finite": finite,
        "max_absolute": absolute,
        "scaled": absolute / scale if finite else None,
        "nonfinite_indices": {
            "left": _nonfinite(left),
            "right": _nonfinite(right),
            "difference": _nonfinite(difference),
        },
    }


def observation_error(candidate, reference, noise_sigma, *, max_error=0.01, max_q=0.1):
    """Measure residuals using a diagonal noise covariance, preserving failures."""
    candidate, reference, noise = map(_array, (candidate, reference, noise_sigma))
    if candidate.shape != reference.shape or not candidate.size:
        raise ValueError("Spectra must have matching nonempty shapes")
    if noise.shape not in ((), candidate.shape):
        raise ValueError("Noise must be scalar or match the spectrum shape")
    if not np.all(np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("Noise must be finite and positive")
    max_error, max_q = _threshold(max_error, "max_error"), _threshold(max_q, "max_q")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = (candidate - reference) / noise
        squared_residual = residual**2
        q = float(np.sum(squared_residual))
    finite = bool(
        np.all(np.isfinite(candidate))
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(residual))
        and np.isfinite(q)
    )
    maximum = float(np.max(np.abs(residual))) if finite else None
    return {
        "finite": finite,
        "passed": finite and maximum <= max_error and q <= max_q,
        "max_error_in_noise": maximum,
        "q": q if finite else None,
        "nonfinite_q": not bool(np.isfinite(q)),
        "maximum_error_index": (
            [
                int(index)
                for index in np.unravel_index(
                    np.argmax(np.abs(residual)), residual.shape
                )
            ]
            if finite
            else None
        ),
        "nonfinite_indices": {
            "candidate": _nonfinite(candidate),
            "reference": _nonfinite(reference),
            "residual": _nonfinite(residual),
            "squared_residual": _nonfinite(squared_residual),
        },
    }


def directional_check(
    function,
    position,
    direction,
    *,
    steps=(1e-2, 1e-3, 1e-4, 1e-5),
    tolerance=1e-3,
    in_domain=None,
    region=None,
    stencil_resolved=None,
):
    """Compare JAX JVP with central differences at two adjacent supplied steps.

    ``in_domain`` rejects endpoints outside the allowed physical/prior domain.
    ``region`` returns scalar/array labels for smooth regions, such as per-layer
    temperature clipping states and RV interpolation cells. A region change
    gives one-sided diagnostics only; excluded steps cannot bridge a pair of
    passing smooth steps. Neither
    finite sampling nor this check guarantees accuracy over an entire prior.
    ``stencil_resolved(center, plus, minus)`` can additionally reject steps lost
    to rounding in transformed model coordinates.
    """
    import jax
    import jax.numpy as jnp

    position, direction, steps = map(_array, (position, direction, steps))
    if position.shape != direction.shape or not position.size:
        raise ValueError("Position and direction must have matching nonempty shapes")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(direction)):
        raise ValueError("Position and direction must be finite")
    if not np.any(direction):
        raise ValueError("Direction must be nonzero")
    if (
        steps.ndim != 1
        or not steps.size
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0)
        or np.any(np.diff(steps) >= 0)
    ):
        raise ValueError("Steps must be positive and strictly decreasing")
    tolerance = _threshold(tolerance, "tolerance")
    domain = in_domain if in_domain is not None else lambda point: True
    result = {
        "passed": False,
        "reason": None,
        "ad_directional_derivative": None,
        "nonfinite_ad_indices": [],
        "steps": [],
    }
    if not bool(domain(position)):
        result["reason"] = "center_out_of_domain"
        return result
    center, ad = jax.jvp(function, (jnp.asarray(position),), (jnp.asarray(direction),))
    center, ad = _array(center), _array(ad)
    if not center.size or center.shape != ad.shape:
        raise ValueError("Function must return a nonempty scalar or array")
    result["ad_directional_derivative"] = _json_array(ad)
    result["nonfinite_ad_indices"] = _nonfinite(ad)
    result["nonfinite_center_indices"] = _nonfinite(center)
    center_finite = bool(np.all(np.isfinite(center)) and np.all(np.isfinite(ad)))
    center_region = region(position) if region is not None else None
    active = direction != 0
    represented_position = np.asarray(jnp.asarray(position))
    previous_passed = False
    for step in steps:
        record = {"step": float(step), "status": None, "passed": False}
        result["steps"].append(record)
        plus, minus = position + step * direction, position - step * direction
        if not bool(domain(plus)) or not bool(domain(minus)):
            record["status"] = "out_of_domain"
            previous_passed = False
            continue
        resolved = all(
            np.all(
                (np.asarray(jnp.asarray(endpoint)) != represented_position)[active]
            )
            for endpoint in (plus, minus)
        )
        if not resolved or (
            stencil_resolved is not None
            and not stencil_resolved(position, plus, minus)
        ):
            record["status"] = "roundoff_limited"
            previous_passed = False
            continue
        plus_value, minus_value = _array(function(plus)), _array(function(minus))
        if plus_value.shape != center.shape or minus_value.shape != center.shape:
            raise ValueError("Function output shape changed across a difference step")
        kink = region is not None and (
            not np.array_equal(region(plus), center_region)
            or not np.array_equal(region(minus), center_region)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            central = (plus_value - minus_value) / (2 * step)
        comparison = derivative_difference(ad, central)
        finite = bool(
            center_finite
            and comparison["finite"]
            and np.all(np.isfinite(plus_value))
            and np.all(np.isfinite(minus_value))
        )
        record.update(
            status="kink" if kink else "smooth" if finite else "nonfinite",
            finite=finite,
            absolute_error=comparison["max_absolute"],
            scaled_error=comparison["scaled"],
            nonfinite_indices={
                "plus": _nonfinite(plus_value),
                "minus": _nonfinite(minus_value),
                "central": _nonfinite(central),
            },
        )
        if kink:
            with np.errstate(over="ignore", invalid="ignore"):
                backward = (center - minus_value) / step
                forward = (plus_value - center) / step
            record["one_sided_derivatives"] = {
                "backward": _json_array(backward),
                "forward": _json_array(forward),
            }
            record["finite"] = bool(
                finite
                and np.all(np.isfinite(backward))
                and np.all(np.isfinite(forward))
            )
        else:
            record["central_derivative"] = _json_array(central)
            record["passed"] = bool(finite and comparison["scaled"] <= tolerance)
        if previous_passed and record["passed"]:
            result["passed"] = True
        previous_passed = record["passed"]
    if not center_finite or any(
        record.get("finite") is False for record in result["steps"]
    ):
        result.update(passed=False, reason="nonfinite_evaluation")
    elif not result["passed"]:
        result["reason"] = (
            "roundoff_limited"
            if any(record["status"] == "roundoff_limited" for record in result["steps"])
            else "no_adjacent_passing_smooth_steps"
        )
    return result
