"""Pure JAX operations for diffgrid opacity interpolation."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import jit, vmap


Array = Union[jnp.ndarray, float]


def cubic_hermite_interpolation(
    x: Array,
    x0: Array,
    x1: Array,
    f0: Array,
    f1: Array,
    g0: Array,
    g1: Array,
) -> jnp.ndarray:
    """Interpolate values and first derivatives on one interval.

    Args:
        x: Evaluation coordinate.
        x0: Lower interval coordinate.
        x1: Upper interval coordinate.
        f0: Value at ``x0``.
        f1: Value at ``x1``.
        g0: Derivative with respect to ``x`` at ``x0``.
        g1: Derivative with respect to ``x`` at ``x1``.

    Returns:
        Cubic Hermite interpolation at ``x``.
    """
    interval = x1 - x0
    fraction = (x - x0) / interval
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction
    h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
    h10 = fraction_cubed - 2.0 * fraction_squared + fraction
    h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
    h11 = fraction_cubed - fraction_squared
    return h00 * f0 + h10 * interval * g0 + h01 * f1 + h11 * interval * g1


def _interpolate_layer_log_cross_section(
    inverse_temperature: jnp.ndarray,
    inverse_temperature_grid: jnp.ndarray,
    log_cross_section_grid: jnp.ndarray,
    log_cross_section_derivative_grid: jnp.ndarray,
) -> jnp.ndarray:
    upper_index = jnp.searchsorted(
        inverse_temperature_grid, inverse_temperature, side="right"
    )
    lower_index = jnp.clip(upper_index - 1, 0, inverse_temperature_grid.size - 2)
    upper_index = lower_index + 1
    return cubic_hermite_interpolation(
        inverse_temperature,
        inverse_temperature_grid[lower_index],
        inverse_temperature_grid[upper_index],
        log_cross_section_grid[lower_index],
        log_cross_section_grid[upper_index],
        log_cross_section_derivative_grid[lower_index],
        log_cross_section_derivative_grid[upper_index],
    )


@jit
def interpolate_log_cross_section(
    temperature: jnp.ndarray,
    inverse_temperature_grid: jnp.ndarray,
    log_cross_section_grid: jnp.ndarray,
    log_cross_section_derivative_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate the layer-aligned log cross-section matrix.

    Args:
        temperature: Layer temperatures in K, shape ``(nlayer,)``.
        inverse_temperature_grid: Inverse-temperature nodes, shape
            ``(ntemperature,)``.
        log_cross_section_grid: Table values, shape
            ``(nlayer, ntemperature, nnu)``.
        log_cross_section_derivative_grid: Table derivatives with the same shape
            as ``log_cross_section_grid``.

    Returns:
        Interpolated log cross sections, shape ``(nlayer, nnu)``.
    """
    inverse_temperature = 1.0 / temperature
    return vmap(
        _interpolate_layer_log_cross_section,
        in_axes=(0, None, 0, 0),
    )(
        inverse_temperature,
        inverse_temperature_grid,
        log_cross_section_grid,
        log_cross_section_derivative_grid,
    )


@jit
def cross_section_matrix(
    temperature: jnp.ndarray,
    inverse_temperature_grid: jnp.ndarray,
    log_cross_section_grid: jnp.ndarray,
    log_cross_section_derivative_grid: jnp.ndarray,
    log_cross_section_floor: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a finite cross-section matrix from a diffgrid table."""
    log_cross_section = interpolate_log_cross_section(
        temperature,
        inverse_temperature_grid,
        log_cross_section_grid,
        log_cross_section_derivative_grid,
    )
    dtype = log_cross_section.dtype
    log_cross_section_ceiling = jnp.log(jnp.finfo(dtype).max) - jnp.log(
        jnp.asarray(2.0, dtype=dtype)
    )
    log_cross_section = jnp.clip(
        log_cross_section,
        jnp.asarray(log_cross_section_floor, dtype=dtype),
        log_cross_section_ceiling,
    )
    xsmatrix = jnp.exp(log_cross_section)

    inverse_temperature = 1.0 / temperature
    coordinate_tolerance = 8.0 * jnp.finfo(inverse_temperature_grid.dtype).eps
    valid = (
        jnp.isfinite(temperature)
        & (temperature > 0.0)
        & (
            inverse_temperature
            >= inverse_temperature_grid[0] * (1.0 - coordinate_tolerance)
        )
        & (
            inverse_temperature
            <= inverse_temperature_grid[-1] * (1.0 + coordinate_tolerance)
        )
    )
    return jnp.where(valid[:, None], xsmatrix, jnp.nan)
