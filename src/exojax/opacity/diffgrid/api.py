"""API for pressure-layer aligned temperature Hermite opacity grids."""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity.base import OpaCalc
from exojax.opacity.diffgrid.contracts import DiffgridInfo
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
        aux: User-specified auxiliary metadata restored from a saved archive.
        user_meta: User provenance metadata restored from a saved archive.
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
        self.aux = {}
        self.user_meta = {}

        for attribute in ("wavelength_order", "wav", "resolution", "molmass"):
            if hasattr(base_opa, attribute):
                setattr(self, attribute, getattr(base_opa, attribute))

        diffgrid_info = build_diffgrid_info(
            base_opa,
            temperature_grid,
            pressure_grid,
            min_cross_section=min_cross_section,
        )
        self._apply_diffgrid_info(diffgrid_info)
        self.ready = True

    @classmethod
    def from_saved_opa(
        cls,
        path: str,
        *,
        strict: bool = True,
        allow_downgrade: bool = False,
    ) -> "OpaDiffgrid":
        """Restore a diffgrid calculator without its teacher or line database.

        Args:
            path: NPZ or Zarr diffgrid archive path.
            strict: Require the current ExoJAX version and exact dtypes for the
                device-resident Diffgrid table state. Set to ``False`` to
                permit compatible dtype conversion.
            allow_downgrade: Permit loading a different common opacity archive
                schema version. The Diffgrid-specific schema must still match.

        Returns:
            A ready-to-use pressure-aligned diffgrid calculator.
        """
        from exojax.opacity.diffgrid.io import load_diffgrid_payload

        arrays, meta = load_diffgrid_payload(
            path,
            strict=strict,
            allow_downgrade=allow_downgrade,
        )
        obj = cls.__new__(cls)
        obj._init_from_saved_payload(arrays, meta, strict=strict)
        return obj

    def _init_from_saved_payload(self, arrays, meta, *, strict: bool) -> None:
        """Rebuild calculator state from a validated archive payload."""
        nu_grid = np.asarray(arrays["nu_grid"]).copy()
        self._validate_active_nu_grid(nu_grid)
        OpaCalc.__init__(self, nu_grid)
        self.method = "diffgrid"

        state = meta["opa_state"]
        self.teacher_method = state.get("teacher_method")
        optional_attributes = state.get("optional_attributes", {})
        for attribute in ("wavelength_order", "resolution", "molmass"):
            if attribute in optional_attributes:
                setattr(self, attribute, optional_attributes[attribute])
        if "wav" in arrays:
            self.wav = np.asarray(arrays["wav"]).copy()

        self.aux = dict(meta.get("aux", {}))
        self.user_meta = dict(meta.get("user_meta", {}))

        field_names = (
            "pressure_grid",
            "temperature_grid",
            "inverse_temperature_grid",
            "log_cross_section_grid",
            "log_cross_section_derivative_grid",
            "log_cross_section_floor",
        )
        converted = {
            name: self._as_active_jax_array(name, arrays[name], strict)
            for name in field_names
        }
        self._validate_active_saved_arrays(converted)
        diffgrid_info = DiffgridInfo(**converted)
        self._apply_diffgrid_info(diffgrid_info)
        self.ready = True

    @staticmethod
    def _as_active_jax_array(name, values, strict: bool) -> jnp.ndarray:
        """Convert a saved array while enforcing the strict dtype policy."""
        host_array = np.asarray(values)
        active_array = jnp.asarray(host_array)
        if strict and np.dtype(active_array.dtype) != host_array.dtype:
            raise ValueError(
                f"Saved diffgrid array '{name}' has dtype {host_array.dtype}, "
                f"but active JAX converts it to {active_array.dtype}. Enable "
                "matching JAX precision or pass strict=False to allow a "
                "compatible conversion."
            )
        return active_array

    @staticmethod
    def _validate_active_nu_grid(nu_grid: np.ndarray) -> None:
        """Ensure the spectral coordinate survives active JAX conversion."""
        active_nu_grid = np.asarray(jnp.asarray(nu_grid))
        if not np.all(np.isfinite(active_nu_grid)) or np.any(
            active_nu_grid <= 0.0
        ):
            raise ValueError(
                "nu_grid must remain finite and positive in the active JAX "
                "dtype."
            )
        if np.unique(active_nu_grid).size != active_nu_grid.size:
            raise ValueError(
                "nu_grid values must remain distinct in the active JAX dtype."
            )

    @staticmethod
    def _validate_active_saved_arrays(arrays) -> None:
        """Check invariants again after conversion to the active JAX dtype."""
        active = {name: np.asarray(value) for name, value in arrays.items()}
        pressure_grid = active["pressure_grid"]
        temperature_grid = active["temperature_grid"]
        inverse_temperature_grid = active["inverse_temperature_grid"]
        value_grid = active["log_cross_section_grid"]
        derivative_grid = active["log_cross_section_derivative_grid"]
        log_floor = active["log_cross_section_floor"]

        for name, coordinate in (
            ("pressure_grid", pressure_grid),
            ("temperature_grid", temperature_grid),
            ("inverse_temperature_grid", inverse_temperature_grid),
        ):
            if not np.all(np.isfinite(coordinate)) or np.any(coordinate <= 0.0):
                raise ValueError(
                    f"{name} must remain finite and positive in the active "
                    "JAX dtype."
                )
        if np.any(np.diff(inverse_temperature_grid) <= 0.0):
            raise ValueError(
                "inverse_temperature_grid must remain strictly increasing in "
                "the active JAX dtype."
            )
        if np.any(np.diff(temperature_grid) >= 0.0):
            raise ValueError(
                "temperature_grid nodes must remain distinct and strictly "
                "decreasing in the active JAX dtype."
            )

        expected_inverse_temperature = 1.0 / temperature_grid
        coordinate_epsilons = [
            np.finfo(coordinate.dtype).eps
            for coordinate in (temperature_grid, inverse_temperature_grid)
            if np.issubdtype(coordinate.dtype, np.floating)
        ]
        tolerance = 8.0 * max(
            coordinate_epsilons,
            default=np.finfo(np.float64).eps,
        )
        if not np.allclose(
            inverse_temperature_grid,
            expected_inverse_temperature,
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError(
                "temperature_grid and inverse_temperature_grid become "
                "inconsistent in the active JAX dtype."
            )

        if not np.all(np.isfinite(value_grid)) or not np.all(
            np.isfinite(derivative_grid)
        ):
            raise ValueError(
                "Diffgrid table values must remain finite in the active JAX "
                "dtype."
            )
        if not np.isfinite(log_floor).item():
            raise ValueError(
                "log_cross_section_floor must remain finite in the active JAX "
                "dtype."
            )
        table_dtype = value_grid.dtype
        if log_floor.item() < np.log(np.finfo(table_dtype).tiny):
            raise ValueError(
                "The saved cross-section floor is below the smallest normal "
                "value in the active JAX dtype."
            )
        if log_floor.item() > np.log(np.finfo(table_dtype).max):
            raise ValueError(
                "The saved cross-section floor exceeds the largest finite "
                "value in the active JAX dtype."
            )

    def _apply_diffgrid_info(self, diffgrid_info: DiffgridInfo) -> None:
        """Attach a validated diffgrid table and rebuild host-side caches."""
        self.diffgrid_info = diffgrid_info
        self.opainfo = diffgrid_info
        self.pressure_grid = diffgrid_info.pressure_grid
        self.temperature_grid = diffgrid_info.temperature_grid
        self.inverse_temperature_grid = diffgrid_info.inverse_temperature_grid
        self.log_cross_section_grid = diffgrid_info.log_cross_section_grid
        self.log_cross_section_derivative_grid = (
            diffgrid_info.log_cross_section_derivative_grid
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
