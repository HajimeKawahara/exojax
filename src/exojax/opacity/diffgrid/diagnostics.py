"""Build-time diagnostics for Diffgrid temperature interpolation.

The functions in this module report numerical errors only. Accuracy
thresholds, pass/fail decisions, and archive provenance belong to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DiffgridComparison:
    """Scalar summary of one Diffgrid-to-teacher comparison.

    Attributes:
        quantiles: Requested quantile levels in caller-supplied order.
        absolute_log_cross_section_error_quantiles: Absolute natural-log
            cross-section errors at ``quantiles``.
        maximum_absolute_log_cross_section_error: Largest absolute natural-log
            cross-section error.
        maximum_error_layer_index: Layer index of the largest error.
        maximum_error_wavenumber_index: Wavenumber index of the largest error.
    """

    quantiles: tuple[float, ...]
    absolute_log_cross_section_error_quantiles: tuple[float, ...]
    maximum_absolute_log_cross_section_error: float
    maximum_error_layer_index: int
    maximum_error_wavenumber_index: int


@jax.jit
def _absolute_log_cross_section_error(
    diffgrid_cross_section,
    teacher_cross_section,
    log_cross_section_floor,
):
    """Return the pointwise error and source finiteness flags."""

    cross_section_floor = jnp.exp(log_cross_section_floor)
    diffgrid_is_finite = jnp.all(jnp.isfinite(diffgrid_cross_section))
    teacher_is_finite = jnp.all(jnp.isfinite(teacher_cross_section))
    error = jnp.abs(
        jnp.log(jnp.maximum(diffgrid_cross_section, cross_section_floor))
        - jnp.log(jnp.maximum(teacher_cross_section, cross_section_floor))
    )
    return error, diffgrid_is_finite, teacher_is_finite


def diffgrid_interval_midpoint_temperatures(diffgrid) -> np.ndarray:
    """Return temperatures at all stored inverse-temperature midpoints.

    The returned temperatures follow the order of the strictly increasing
    ``diffgrid.inverse_temperature_grid``. No assumption of uniform spacing is
    made.

    Raises:
        ValueError: If the Diffgrid object is not ready or its stored
            inverse-temperature grid is invalid.
    """

    if hasattr(diffgrid, "ready") and not diffgrid.ready:
        raise ValueError("diffgrid must be ready before it can be diagnosed.")
    if not hasattr(diffgrid, "inverse_temperature_grid"):
        raise ValueError("diffgrid must provide inverse_temperature_grid.")

    inverse_temperature_grid = np.asarray(diffgrid.inverse_temperature_grid)
    if (
        inverse_temperature_grid.ndim != 1
        or inverse_temperature_grid.size < 2
    ):
        raise ValueError(
            "diffgrid inverse_temperature_grid must contain at least two "
            "one-dimensional nodes."
        )
    if (
        not np.all(np.isfinite(inverse_temperature_grid))
        or np.any(inverse_temperature_grid <= 0.0)
        or np.any(np.diff(inverse_temperature_grid) <= 0.0)
    ):
        raise ValueError(
            "diffgrid inverse_temperature_grid must be finite, positive, and "
            "strictly increasing."
        )

    inverse_temperature_midpoints = 0.5 * (
        inverse_temperature_grid[:-1] + inverse_temperature_grid[1:]
    )
    return 1.0 / inverse_temperature_midpoints


def _validated_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    quantile_array = np.asarray(quantiles, dtype=float)
    if quantile_array.ndim != 1 or quantile_array.size == 0:
        raise ValueError(
            "quantiles must be a non-empty one-dimensional sequence."
        )
    if not np.all(np.isfinite(quantile_array)) or np.any(
        (quantile_array < 0.0) | (quantile_array > 1.0)
    ):
        raise ValueError("quantiles must contain finite values in [0, 1].")
    return tuple(float(value) for value in quantile_array)


def _validated_comparison_grids(diffgrid, teacher):
    if hasattr(diffgrid, "ready") and not diffgrid.ready:
        raise ValueError("diffgrid must be ready before it can be diagnosed.")
    if not hasattr(diffgrid, "xsmatrix") or not hasattr(diffgrid, "nu_grid"):
        raise ValueError("diffgrid must provide nu_grid and xsmatrix.")
    if hasattr(teacher, "ready") and not teacher.ready:
        raise ValueError("teacher must be ready before it can be diagnosed.")
    if not hasattr(teacher, "xsmatrix") or not hasattr(teacher, "nu_grid"):
        raise ValueError("teacher must provide nu_grid and xsmatrix.")
    if not hasattr(diffgrid, "pressure_grid"):
        raise ValueError("diffgrid must provide pressure_grid.")

    diffgrid_nu_grid = np.asarray(diffgrid.nu_grid)
    teacher_nu_grid = np.asarray(teacher.nu_grid)
    if (
        diffgrid_nu_grid.ndim != 1
        or diffgrid_nu_grid.size == 0
        or not np.all(np.isfinite(diffgrid_nu_grid))
        or np.any(diffgrid_nu_grid <= 0.0)
        or teacher_nu_grid.shape != diffgrid_nu_grid.shape
        or not np.array_equal(teacher_nu_grid, diffgrid_nu_grid)
    ):
        raise ValueError(
            "teacher and diffgrid wavenumber grids must match point by point."
        )

    pressure_grid = np.asarray(diffgrid.pressure_grid)
    if (
        pressure_grid.ndim != 1
        or pressure_grid.size == 0
        or not np.all(np.isfinite(pressure_grid))
        or np.any(pressure_grid <= 0.0)
    ):
        raise ValueError(
            "diffgrid pressure_grid must be finite, positive, and non-empty."
        )
    return pressure_grid, diffgrid_nu_grid


def _validated_log_cross_section_floor(diffgrid):
    diffgrid_info = getattr(diffgrid, "diffgrid_info", None)
    if diffgrid_info is not None:
        log_cross_section_floor = (
            diffgrid_info.log_cross_section_floor
        )
    elif hasattr(diffgrid, "log_cross_section_floor"):
        log_cross_section_floor = diffgrid.log_cross_section_floor
    else:
        raise ValueError("diffgrid must provide log_cross_section_floor.")
    log_floor = jnp.asarray(log_cross_section_floor)
    if log_floor.ndim != 0:
        raise ValueError("diffgrid log_cross_section_floor must be scalar.")
    active_floor = np.asarray(jnp.exp(log_floor))
    if not np.isfinite(active_floor).item() or active_floor.item() <= 0.0:
        raise ValueError(
            "diffgrid log_cross_section_floor must represent a finite, "
            "positive value in the active JAX dtype."
        )
    return log_floor


def compare_diffgrid_with_teacher(
    diffgrid,
    teacher,
    temperature_profile,
    quantiles: Sequence[float] = (0.99,),
) -> DiffgridComparison:
    """Compare one Diffgrid profile with its reference opacity calculator.

    The comparison uses the pressure and cross-section floor stored by
    ``diffgrid``. It computes

    ``abs(log(max(diffgrid_xs, floor)) - log(max(teacher_xs, floor)))``

    at every layer and wavenumber. This function reports metrics but does not
    apply an accuracy threshold. Quantiles combine all layer-wavenumber
    entries.
    This is a host-side build-time diagnostic, not a JIT-compatible model
    operation. It transfers one pointwise error matrix to the host for exact
    NumPy quantiles.

    Args:
        diffgrid: Ready pressure-aligned Diffgrid opacity calculator.
        teacher: Ready reference opacity calculator on the same wavenumber
            grid. Its ``xsmatrix`` method must accept temperature and pressure
            profiles.
        temperature_profile: Layer temperatures in K, shape ``(nlayer,)``.
        quantiles: Quantile levels to report. Defaults to ``(0.99,)``.

    Returns:
        A :class:`DiffgridComparison` containing host scalar metrics.

    Raises:
        ValueError: If the calculators, grids, floor, profile, or requested
            quantiles are incompatible.
        FloatingPointError: If either calculator returns non-finite cross
            sections.
    """

    quantiles = _validated_quantiles(quantiles)
    pressure_grid, nu_grid = _validated_comparison_grids(diffgrid, teacher)
    log_cross_section_floor = _validated_log_cross_section_floor(diffgrid)

    temperature_profile = np.asarray(temperature_profile)
    if (
        temperature_profile.ndim != 1
        or temperature_profile.shape != pressure_grid.shape
    ):
        raise ValueError(
            "temperature_profile shape must match diffgrid pressure_grid."
        )
    if not np.all(np.isfinite(temperature_profile)) or np.any(
        temperature_profile <= 0.0
    ):
        raise ValueError("temperature_profile must be finite and positive.")

    expected_shape = (pressure_grid.size, nu_grid.size)
    diffgrid_cross_section = jnp.asarray(
        diffgrid.xsmatrix(temperature_profile)
    )
    if diffgrid_cross_section.shape != expected_shape:
        raise ValueError(
            "Diffgrid cross-section matrix must have shape "
            f"{expected_shape}; got {diffgrid_cross_section.shape}."
        )
    teacher_cross_section = jnp.asarray(
        teacher.xsmatrix(temperature_profile, pressure_grid)
    )
    if teacher_cross_section.shape != expected_shape:
        raise ValueError(
            "Teacher cross-section matrix must have shape "
            f"{expected_shape}; got {teacher_cross_section.shape}."
        )

    error_device, diffgrid_is_finite, teacher_is_finite = (
        _absolute_log_cross_section_error(
            diffgrid_cross_section,
            teacher_cross_section,
            log_cross_section_floor,
        )
    )
    del diffgrid_cross_section, teacher_cross_section
    if not bool(np.asarray(diffgrid_is_finite)):
        raise FloatingPointError(
            "Diffgrid produced non-finite cross sections."
        )
    if not bool(np.asarray(teacher_is_finite)):
        raise FloatingPointError("Teacher produced non-finite cross sections.")
    error = np.array(error_device)
    del error_device

    flat_maximum_index = int(np.argmax(error))
    maximum_error = float(error.ravel()[flat_maximum_index])
    layer_index, wavenumber_index = np.unravel_index(
        flat_maximum_index, error.shape
    )
    quantile_errors = np.quantile(
        error,
        np.asarray(quantiles),
        overwrite_input=True,
    )
    return DiffgridComparison(
        quantiles=quantiles,
        absolute_log_cross_section_error_quantiles=tuple(
            float(value) for value in np.atleast_1d(quantile_errors)
        ),
        maximum_absolute_log_cross_section_error=maximum_error,
        maximum_error_layer_index=int(layer_index),
        maximum_error_wavenumber_index=int(wavenumber_index),
    )


__all__ = [
    "DiffgridComparison",
    "compare_diffgrid_with_teacher",
    "diffgrid_interval_midpoint_temperatures",
]
