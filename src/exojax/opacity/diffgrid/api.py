"""API for pressure-layer aligned temperature Hermite opacity grids."""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity.base import OpaCalc
from exojax.opacity.diffgrid.core import cross_section_matrix
from exojax.opacity.diffgrid.precompute import build_diffgrid_info


class OpaDiffgrid(OpaCalc):
    """Opacity calculator using a pressure-aligned temperature Hermite grid.

    The table stores log cross sections and their derivatives with respect to
    inverse temperature. Pressure is fixed to the atmospheric layer grid used at
    construction time and is never interpolated.

    Attributes:
        method: Always ``"diffgrid"`` for this calculator.
        teacher_method: Method name of the calculator used to build the table.
        diffgrid_info: Immutable table values and coordinates.
    """

    def __init__(
        self,
        base_opa,
        temperature_grid: Union[np.ndarray, jnp.ndarray],
        pressure_grid: Union[np.ndarray, jnp.ndarray],
        min_cross_section: float = 1.0e-35,
    ) -> None:
        """Initialize and build a diffgrid opacity table.

        Args:
            base_opa: Differentiable opacity calculator providing ``xsmatrix``.
                ``OpaPremodit`` is the standard teacher.
            temperature_grid: Temperature nodes in K.
            pressure_grid: Fixed pressure assigned to each atmospheric layer in
                bar.
            min_cross_section: Positive floor applied before taking logarithms.
                Defaults to ``1e-35`` cm2 to stabilize zero or negative
                round-off from the teacher.

        Raises:
            ValueError: If a grid is invalid or the teacher is not ready.
        """
        self._validate_teacher(base_opa)
        temperature_grid = self._validated_grid(
            "temperature_grid", temperature_grid, minimum_size=2
        )
        pressure_grid = self._validated_grid(
            "pressure_grid", pressure_grid, minimum_size=1
        )
        if np.unique(temperature_grid).size != temperature_grid.size:
            raise ValueError("temperature_grid values must be unique.")
        if (
            not np.isfinite(min_cross_section) or min_cross_section <= 0.0
        ):
            raise ValueError("min_cross_section must be finite and positive.")

        super().__init__(base_opa.nu_grid)
        self.method = "diffgrid"
        self.teacher_method = getattr(base_opa, "method", None)

        for attribute in ("wavelength_order", "wav", "resolution", "molmass"):
            if hasattr(base_opa, attribute):
                setattr(self, attribute, getattr(base_opa, attribute))

        self.diffgrid_info = build_diffgrid_info(
            base_opa,
            temperature_grid,
            pressure_grid,
            min_cross_section=min_cross_section,
        )
        self.opainfo = self.diffgrid_info
        self.pressure_grid = self.diffgrid_info.pressure_grid
        self.temperature_grid = self.diffgrid_info.temperature_grid
        self.inverse_temperature_grid = (
            self.diffgrid_info.inverse_temperature_grid
        )
        self.log_cross_section_grid = self.diffgrid_info.log_cross_section_grid
        self.log_cross_section_derivative_grid = (
            self.diffgrid_info.log_cross_section_derivative_grid
        )
        self._pressure_grid_host = np.asarray(self.pressure_grid).copy()
        self._inverse_temperature_grid_host = np.asarray(
            self.inverse_temperature_grid
        ).copy()
        temperature_grid_host = np.asarray(self.temperature_grid)
        self._temperature_range_host = (
            float(np.min(temperature_grid_host)),
            float(np.max(temperature_grid_host)),
        )
        self.ready = True

    @staticmethod
    def _validate_teacher(base_opa) -> None:
        if not hasattr(base_opa, "nu_grid") or not hasattr(base_opa, "xsmatrix"):
            raise ValueError("base_opa must provide nu_grid and xsmatrix.")
        if hasattr(base_opa, "ready") and not base_opa.ready:
            raise ValueError(
                "base_opa must be ready before building a diffgrid table."
            )

    @staticmethod
    def _validated_grid(name, values, minimum_size: int) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if values.size < minimum_size:
            raise ValueError(f"{name} must contain at least {minimum_size} values.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} values must be finite and positive.")
        return values

    def _raise_for_pressure_mismatch(self, pressure_grid) -> None:
        pressure_grid = np.asarray(pressure_grid)
        if not np.allclose(
            pressure_grid,
            self._pressure_grid_host,
            rtol=1.0e-7,
            atol=0.0,
        ):
            raise ValueError(
                "pressure_grid does not match the pressure layers used to build "
                "this diffgrid table; rebuild the table for the new pressures."
            )

    def check_pressure_grid(
        self, pressure_grid: Union[np.ndarray, jnp.ndarray]
    ) -> None:
        """Raise an error unless pressure matches the table layer by layer."""
        if isinstance(pressure_grid, jax.core.Tracer):
            raise ValueError(
                "pressure_grid is fixed for OpaDiffgrid and cannot be a traced "
                "argument; validate it before JIT and omit pressure in compiled "
                "calls."
            )
        pressure_grid = np.asarray(pressure_grid)
        if (
            pressure_grid.ndim != 1
            or pressure_grid.shape != self.pressure_grid.shape
        ):
            raise ValueError(
                "pressure_grid shape does not match the pressure layers used to "
                "build this diffgrid table."
            )
        self._raise_for_pressure_mismatch(pressure_grid)

    def _check_temperature_grid_if_concrete(self, temperature) -> None:
        if isinstance(temperature, jax.core.Tracer):
            return
        temperature = np.asarray(temperature)
        temperature_min, temperature_max = self._temperature_range_host
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError(
                "temperature values must be finite and within the diffgrid "
                f"range [{temperature_min}, {temperature_max}] K."
            )
        inverse_temperature = 1.0 / temperature
        if (
            np.any(inverse_temperature < self._inverse_temperature_grid_host[0])
            or np.any(
                inverse_temperature > self._inverse_temperature_grid_host[-1]
            )
        ):
            raise ValueError(
                "temperature values must be finite and within the diffgrid "
                f"range [{temperature_min}, {temperature_max}] K."
            )

    def xsmatrix(
        self,
        Tarr: Union[np.ndarray, jnp.ndarray],
        Parr: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Compute the layer-aligned cross-section matrix.

        Args:
            Tarr: Layer temperatures in K, shape ``(nlayer,)``.
            Parr: Optional pressure profile in bar. If supplied, it must
                match the profile used to build the table. It may be omitted in
                compiled models because pressure is fixed by construction.

        Returns:
            Cross-section matrix in cm2, shape ``(nlayer, nnu)``.
        """
        Tarr = jnp.asarray(Tarr)
        if Tarr.ndim != 1 or Tarr.shape != self.pressure_grid.shape:
            raise ValueError(
                "temperature shape must match the pressure layers used to build "
                "this diffgrid table."
            )
        if Parr is not None:
            self.check_pressure_grid(Parr)
        self._check_temperature_grid_if_concrete(Tarr)
        return cross_section_matrix(
            Tarr,
            self.diffgrid_info.inverse_temperature_grid,
            self.diffgrid_info.log_cross_section_grid,
            self.diffgrid_info.log_cross_section_derivative_grid,
            self.diffgrid_info.log_cross_section_floor,
        )

    def xsvector(self, T: float, P: float) -> jnp.ndarray:
        """Reject scalar evaluation because diffgrid is layer aligned."""
        raise NotImplementedError(
            "OpaDiffgrid is pressure-layer aligned; use xsmatrix instead of "
            "xsvector."
        )
