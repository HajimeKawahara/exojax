"""Data contracts for pressure-layer aligned opacity grids."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class DiffgridInfo:
    """Immutable container for a diffgrid opacity table.

    Attributes:
        pressure_grid: Pressure assigned to each atmospheric layer, shape
            ``(nlayer,)``.
        temperature_grid: Temperature nodes ordered to match
            ``inverse_temperature_grid``, shape ``(ntemperature,)``.
        inverse_temperature_grid: Inverse-temperature nodes in ascending order,
            shape ``(ntemperature,)``.
        log_cross_section_grid: Log cross sections, shape
            ``(nlayer, ntemperature, nnu)``.
        log_cross_section_derivative_grid: Derivatives of log cross section with
            respect to inverse temperature, with the same shape as
            ``log_cross_section_grid``.
        log_cross_section_floor: Logarithm of the cross-section floor used when
            building and evaluating the table.
    """

    pressure_grid: jnp.ndarray
    temperature_grid: jnp.ndarray
    inverse_temperature_grid: jnp.ndarray
    log_cross_section_grid: jnp.ndarray
    log_cross_section_derivative_grid: jnp.ndarray
    log_cross_section_floor: jnp.ndarray
