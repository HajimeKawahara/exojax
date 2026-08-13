"""Build diffgrid tables from a differentiable opacity calculator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import jvp, lax

from exojax.opacity.diffgrid.contracts import DiffgridInfo


def build_diffgrid_info(
    base_opa,
    temperature_grid: np.ndarray,
    pressure_grid: np.ndarray,
    min_cross_section: float = 1.0e-35,
) -> DiffgridInfo:
    """Build log cross sections and inverse-temperature derivatives.

    Args:
        base_opa: Differentiable opacity calculator providing ``xsmatrix``.
        temperature_grid: Temperature nodes in K, shape ``(ntemperature,)``.
        pressure_grid: Pressure assigned to each layer in bar, shape
            ``(nlayer,)``.
        min_cross_section: Positive floor applied before taking the logarithm.
            Defaults to ``1e-35`` cm2.

    Returns:
        Pressure-aligned diffgrid table information.
    """
    temperature_grid = np.asarray(temperature_grid)
    sort_index = np.argsort(1.0 / temperature_grid)
    temperature_grid = jnp.asarray(temperature_grid[sort_index])
    temperature_grid_host = np.asarray(temperature_grid)
    if (
        not np.all(np.isfinite(temperature_grid_host))
        or np.any(temperature_grid_host <= 0.0)
    ):
        raise ValueError(
            "temperature_grid values must remain finite and positive in the "
            "active JAX dtype."
        )
    inverse_temperature_grid = 1.0 / temperature_grid
    pressure_grid = jnp.asarray(pressure_grid)
    inverse_temperature_grid_host = np.asarray(inverse_temperature_grid)
    if (
        not np.all(np.isfinite(inverse_temperature_grid_host))
        or np.any(np.diff(inverse_temperature_grid_host) <= 0.0)
    ):
        raise ValueError(
            "temperature_grid values must remain finite and unique in the "
            "active JAX dtype."
        )
    pressure_grid_host = np.asarray(pressure_grid)
    if (
        not np.all(np.isfinite(pressure_grid_host))
        or np.any(pressure_grid_host <= 0.0)
    ):
        raise ValueError(
            "pressure_grid values must remain finite and positive in the "
            "active JAX dtype."
        )

    def log_cross_section_at_temperature(temperature):
        temperature = jnp.full(
            pressure_grid.shape,
            temperature,
            dtype=temperature.dtype,
        )
        cross_section = base_opa.xsmatrix(temperature, pressure_grid)
        floor = jnp.asarray(min_cross_section, dtype=cross_section.dtype)
        return jnp.log(jnp.maximum(cross_section, floor))

    def derivative_at_node(inverse_temperature_node, temperature_node):
        def log_cross_section_at(inverse_temperature):
            temperature = temperature_node / (
                1.0
                + (inverse_temperature - inverse_temperature_node)
                * temperature_node
            )
            return log_cross_section_at_temperature(temperature)

        return jvp(
            log_cross_section_at,
            (inverse_temperature_node,),
            (jnp.ones_like(inverse_temperature_node),),
        )[1]

    def build_temperature_node(node):
        return derivative_at_node(node[0], node[1])

    # Keep node values on the teacher's ordinary path; JVP supplies slopes only.
    log_cross_section_grid = jnp.stack(
        [log_cross_section_at_temperature(node) for node in temperature_grid]
    )
    log_cross_section_derivative_grid = lax.map(
        build_temperature_node,
        (inverse_temperature_grid, temperature_grid),
    )
    log_cross_section_grid = jnp.swapaxes(log_cross_section_grid, 0, 1)
    log_cross_section_derivative_grid = jnp.swapaxes(
        log_cross_section_derivative_grid, 0, 1
    )
    min_cross_section = jnp.asarray(
        min_cross_section, dtype=log_cross_section_grid.dtype
    )
    if (
        not bool(jnp.isfinite(min_cross_section))
        or not bool(
            min_cross_section >= jnp.finfo(log_cross_section_grid.dtype).tiny
        )
    ):
        raise ValueError(
            "min_cross_section must remain finite and at least the smallest "
            "normal value in the active JAX dtype."
        )
    log_cross_section_floor = jnp.log(min_cross_section)

    if not bool(jnp.all(jnp.isfinite(log_cross_section_grid))):
        raise FloatingPointError("Diffgrid log cross sections must be finite.")
    if not bool(jnp.all(jnp.isfinite(log_cross_section_derivative_grid))):
        raise FloatingPointError(
            "Diffgrid log cross-section derivatives must be finite."
        )

    return DiffgridInfo(
        pressure_grid=pressure_grid,
        temperature_grid=temperature_grid,
        inverse_temperature_grid=inverse_temperature_grid,
        log_cross_section_grid=log_cross_section_grid,
        log_cross_section_derivative_grid=log_cross_section_derivative_grid,
        log_cross_section_floor=log_cross_section_floor,
    )
