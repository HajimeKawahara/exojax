"""API for Correlated-K Distribution (CKD) opacity calculations.

This module provides the OpaCKD class for correlated-k distribution opacity calculations.
CKD enables efficient radiative transfer by reducing the spectral dimensionality while
maintaining accuracy through k-distribution statistical representation.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import json

import jax.numpy as jnp
import numpy as np
from jax import vmap

from exojax.opacity.base import OpaCalc
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.core import gauss_legendre_grid
from exojax.opacity.ckd.core import compute_ckd_tables
from exojax.opacity.ckd.core import interpolate_log_k_2d
from exojax.opacity.ckd.io import _hash_json
from exojax.opacity.ckd.io import _base_fingerprint
from exojax.opacity.ckd.io import _ckd_save_as_npz
from exojax.utils.spectral_bands import spectral_bands




class OpaCKD(OpaCalc):
    """Opacity Calculator for Correlated-K Distribution (CKD) method.

    OpaCKD provides efficient radiative transfer calculations by using pre-computed
    k-distribution tables that statistically represent spectral opacity variations.
    This enables fast atmospheric modeling while maintaining accuracy.

    Attributes:
        method: Always "ckd" for this calculator
        base_opa: Underlying opacity calculator (OpaPremodit, OpaModit, etc.)
        Ng: Number of Gauss-Legendre quadrature points
        ckd_info: Pre-computed CKD table information
        ready: Whether calculator is ready for opacity computation
    """

    def __init__(
        self,
        base_opa,
        Ng: int = 32,
        band_width: float = 50.0,
        band_spacing: str = "log",
    ) -> None:
        """Initialize OpaCKD opacity calculator.

        Args:
            base_opa: Base opacity calculator (OpaPremodit, OpaModit, etc.)
            Ng: Number of Gauss-Legendre quadrature points
            band_width: Width of each spectral band (cm⁻¹)
            band_spacing: "linear" or "log" spacing for band generation (default: "log")

        Raises:
            ValueError: If base opacity calculator is not ready or invalid parameters
        """
        if not hasattr(base_opa, "nu_grid"):
            raise ValueError("Base opacity calculator must have nu_grid attribute")

        # Initialize parent with base_opa's grid for compatibility
        super().__init__(base_opa.nu_grid)

        self.method = "ckd"
        self.base_opa = base_opa
        self.Ng = Ng
        self.band_width = band_width
        self.band_spacing = band_spacing

        # Auto-generate spectral bands from base_opa grid
        self._setup_spectral_bands()

        # molmass if available
        self.molmass = None

        # Initialize state
        self.ckd_info = None
        self.ready = False

    @classmethod
    def load_only(cls) -> OpaCKD:
        """headless initialization for loading from saved tables (without base_opa)."""
        self = object.__new__(cls)  # no __init__
        # mimimal initialization
        self.method = "ckd"
        self.base_opa = None
        self.Ng = None
        self.band_width = None
        self.band_spacing = "log"
        self.ckd_info = None
        self.nu_bands = None
        self.band_edges = None
        self.molmass = None
        self.ready = False
        self._expected_base_hash = None  # uses validation when loading
        self._expected_base_meta = None
        # dummy attributes to satisfy OpaCalc
        self.nu_grid = None
        return self

    def _setup_spectral_bands(self) -> None:
        """Set up spectral bands from base opacity grid."""

        # Get spectral range from base opacity calculator
        nu_min = float(self.base_opa.nu_grid[0])
        nu_max = float(self.base_opa.nu_grid[-1])

        # Generate band centers and edges
        nu_bands, band_edges = spectral_bands(
            nu_min=nu_min,
            nu_max=nu_max,
            band_width=self.band_width,
            spacing=self.band_spacing,
        )

        self.nu_bands = jnp.asarray(nu_bands)
        self.band_edges = jnp.asarray(band_edges)

    def _validate_precompute_inputs(
        self, T_grid: jnp.ndarray, P_grid: jnp.ndarray
    ) -> None:
        """Validate inputs for precompute_tables.

        Args:
            T_grid: Temperature grid in Kelvin
            P_grid: Pressure grid in bar

        Raises:
            ValueError: If validation fails
        """
        # Check base opacity calculator
        if not hasattr(self.base_opa, "xsmatrix"):
            raise ValueError("Base opacity calculator must have xsmatrix method")

        # Validate grid dimensions
        if len(T_grid) == 0 or len(P_grid) == 0:
            raise ValueError("T_grid and P_grid must not be empty")

        # Validate physical values
        if jnp.any(T_grid <= 0):
            raise ValueError("All temperatures must be positive")

        if jnp.any(P_grid <= 0):
            raise ValueError("All pressures must be positive")

    def _process_spectral_band(
        self,
        i: int,
        band_edge: jnp.ndarray,
        xsmatrix_full: jnp.ndarray,
        compute_ckd_tables,
    ) -> Optional[jnp.ndarray]:
        """Process a single spectral band for CKD computation.

        Args:
            i: Band index
            band_edge: [left, right] edge positions
            xsmatrix_full: Full cross-section matrix
            compute_ckd_tables: CKD computation function

        Returns:
            CKD results for this band, or None if band has no coverage
        """
        # Extract wavenumber range for this band using edges
        nu_left, nu_right = band_edge[0], band_edge[1]

        # Find indices in base_opa.nu_grid that fall within this band
        mask = (self.base_opa.nu_grid >= nu_left) & (self.base_opa.nu_grid <= nu_right)

        if not jnp.any(mask):
            print(f"  Band {i+1}: No coverage, skipping")
            return None

        # Extract subgrid cross-sections for this band (no expensive xsmatrix call!)
        # Handle both 2D (nT, nnu) and 3D (nT, nP, nnu) cases
        if len(xsmatrix_full.shape) == 3:
            xsmatrix_band = xsmatrix_full[:, :, mask]
        else:
            xsmatrix_band = xsmatrix_full[:, mask]
        n_freq_band = jnp.sum(mask)

        print(
            f"  Band {i+1}: [{nu_left:.1f}, {nu_right:.1f}] cm⁻¹, {n_freq_band} frequencies"
        )

        # Compute CKD for this band
        log_kggrid_band, _, _ = compute_ckd_tables(xsmatrix_band, self.Ng)

        return log_kggrid_band

    def precompute_tables(
        self,
        T_grid: Union[np.ndarray, jnp.ndarray],
        P_grid: Union[np.ndarray, jnp.ndarray],
        *,
        to_path: Optional[str] = None,
        io_format: str = "npz",
        overwrite: bool = False,
    ) -> None:
        """Pre-compute CKD tables for given T,P grids.

        Args:
            T_grid: Temperature grid in Kelvin
            P_grid: Pressure grid in bar
        """
        # Step 1: Setup and validation
        # Convert to JAX arrays
        T_grid = jnp.asarray(T_grid)
        P_grid = jnp.asarray(P_grid)

        self._validate_precompute_inputs(T_grid, P_grid)
        ggrid, weights = gauss_legendre_grid(self.Ng)
        print(
            f"Generated g-grid: {self.Ng} points, range [{ggrid[0]:.4f}, {ggrid[-1]:.4f}]"
        )
        xsmatrix_full = self.base_opa.xsmatrix(T_grid, P_grid)

        # Initialize storage for all bands
        nT, nP = len(T_grid), len(P_grid)
        nnu_bands = len(self.nu_bands)
        log_kggrid = jnp.zeros((nT, nP, self.Ng, nnu_bands))

        # Process each spectral band using precise edges
        print(f"Processing {nnu_bands} spectral bands...")
        for i, band_edge in enumerate(self.band_edges):
            # Process this band
            log_kggrid_band = self._process_spectral_band(
                i, band_edge, xsmatrix_full, compute_ckd_tables
            )

            # Store results if band has coverage
            if log_kggrid_band is not None:
                log_kggrid = log_kggrid.at[:, :, :, i].set(log_kggrid_band)

        # Step 5: Create CKD table info and finalize
        print("Creating CKD table info...")
        self.ckd_info = CKDTableInfo(
            log_kggrid=log_kggrid,
            ggrid=ggrid,
            weights=weights,
            T_grid=T_grid,
            P_grid=P_grid,
            nu_bands=self.nu_bands,
            band_edges=self.band_edges,
        )

        self.ready = True
        print(f"CKD precomputation complete! Ready for interpolation.")
        print(
            f"Table dimensions: T={len(T_grid)}, P={len(P_grid)}, g={self.Ng}, bands={nnu_bands}"
        )
        # Optionally save to file
        if to_path is not None:
            if io_format != "npz":
                raise ValueError(
                    f"Unsupported io_format={io_format}. Only 'npz' is supported for now."
                )
            _ckd_save_as_npz(self, to_path, overwrite=overwrite)
            print(f"Saved CKD table to: {to_path}")

    def save_tables(
        self, path: str, *, io_format: str = "npz", overwrite: bool = False
    ) -> None:
        if not self.ready or self.ckd_info is None:
            raise RuntimeError(
                "CKD table is not prepared. Run precompute_tables first."
            )
        if io_format != "npz":
            raise ValueError(
                f"Unsupported io_format={io_format}. Only 'npz' is supported for now."
            )
        _ckd_save_as_npz(self, path, overwrite=overwrite)

    def _interpolate_log_k(self, T: float, P: float) -> jnp.ndarray:
        """JAX-compatible 2D interpolation of log_kggrid at given T,P.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar

        Returns:
            Interpolated log k-values, shape (Ng, nnu_bands)
        """
        return interpolate_log_k_2d(
            self.ckd_info.log_kggrid, self.ckd_info.T_grid, self.ckd_info.P_grid, T, P
        )

    def xsarray_ckd(self, T: float, P: float) -> jnp.ndarray:
        """Compute CKD cross section array using interpolation.

        Interpolates pre-computed CKD tables at given T,P and returns the 2D array
        with shape (Ng, nnu_bands) containing g-ordinates and spectral bands.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar

        Returns:
            Cross section array in cm², shape (Ng, nnu_bands)
            First dimension: g-ordinates (quadrature points)
            Second dimension: spectral bands

        """
        log_k_interp = self._interpolate_log_k(T, P)  # Shape: (Ng, nnu_bands)
        return jnp.exp(log_k_interp)

    def xstensor_ckd(
        self,
        T_array: Union[np.ndarray, jnp.ndarray],
        P_array: Union[np.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute CKD cross section tensor using interpolation.

        Computes CKD cross-sections for paired (T,P) values: (T1,P1), (T2,P2), ...
        Returns a 3D tensor with layers, g-ordinates, and spectral bands.

        Args:
            T_array: Temperature array in Kelvin, shape (Nlayer,)
            P_array: Pressure array in bar, shape (Nlayer,)

        Returns:
            Cross section tensor in cm², shape (Nlayer, Ng, nnu_bands)
            First dimension: atmospheric layers
            Second dimension: g-ordinates (quadrature points)
            Third dimension: spectral bands

        """
        xsarray_vmap = vmap(self.xsarray_ckd, in_axes=(0, 0))
        return xsarray_vmap(T_array, P_array)

    @staticmethod
    def _load_tables_payload(base_opa, path: str, io_format: str):
        if io_format != "npz":
            raise ValueError("Only npz is supported for now.")

        with np.load(path, allow_pickle=False) as data:
            meta_bytes = np.asarray(data["meta"], dtype=np.uint8)
            meta = json.loads(meta_bytes.tobytes().decode("utf-8"))

            expected_hash = meta.get("base_fingerprint_hash")
            expected_meta = meta.get("base_fingerprint")

            if base_opa is not None:
                actual_fp = _base_fingerprint(base_opa)
                actual_hash = _hash_json(actual_fp)
                if expected_hash is not None and expected_hash != actual_hash:
                    raise ValueError(
                        "Loaded CKD table does not match base_opa fingerprint."
                    )
            else:
                if expected_hash is None:
                    raise ValueError(
                        "Loaded CKD table is missing base fingerprint metadata; provide base_opa to validate."
                    )

            arrays = dict(
                log_kggrid=np.asarray(data["log_kggrid"]),
                ggrid=np.asarray(data["ggrid"]),
                weights=np.asarray(data["weights"]),
                T_grid=np.asarray(data["T_grid"]),
                P_grid=np.asarray(data["P_grid"]),
                nu_bands=np.asarray(data["nu_bands"]),
                band_edges=np.asarray(data["band_edges"]),
            )

        ggrid_np = arrays["ggrid"]
        ggrid_len = ggrid_np.shape[0] if ggrid_np.ndim == 1 else ggrid_np.size
        Ng_meta = int(meta.get("Ng", ggrid_len))
        if ggrid_np.ndim != 1 or ggrid_np.shape[0] != Ng_meta:
            raise ValueError(
                f"Inconsistent Ng between metadata ({Ng_meta}) and g-grid ({ggrid_len})"
            )
        if not np.all(np.isfinite(ggrid_np)):
            raise ValueError("ggrid must contain finite values")
        if not np.all((ggrid_np >= 0.0) & (ggrid_np <= 1.0)):
            raise ValueError("ggrid values must lie within [0, 1]")
        if np.any(np.diff(ggrid_np) <= 0.0):
            raise ValueError("ggrid must be strictly increasing")
        if arrays["weights"].ndim != 1 or arrays["weights"].shape[0] != Ng_meta:
            raise ValueError("weights shape does not match Ng in metadata")
        if not np.all(np.isfinite(arrays["weights"])):
            raise ValueError("weights must contain finite values")
        if not np.all(arrays["weights"] > 0.0):
            raise ValueError("weights must be positive")
        if not np.isclose(np.sum(arrays["weights"]), 1.0, rtol=1.0e-5, atol=1.0e-8):
            raise ValueError("weights must sum to one")

        log_kggrid_np = arrays["log_kggrid"]
        if log_kggrid_np.ndim != 4 or log_kggrid_np.shape[2] != Ng_meta:
            raise ValueError("log_kggrid shape does not match Ng in metadata")
        if not np.all(np.isfinite(log_kggrid_np)):
            raise ValueError("log_kggrid must contain finite values")

        if arrays["T_grid"].ndim != 1 or arrays["P_grid"].ndim != 1:
            raise ValueError("T_grid and P_grid must be one-dimensional")
        if (
            arrays["T_grid"].shape[0] != log_kggrid_np.shape[0]
            or arrays["P_grid"].shape[0] != log_kggrid_np.shape[1]
        ):
            raise ValueError("T_grid or P_grid shape does not match log_kggrid")
        if not np.all(np.isfinite(arrays["T_grid"])) or not np.all(
            np.isfinite(arrays["P_grid"])
        ):
            raise ValueError("T_grid and P_grid must contain finite values")
        if not np.all(arrays["T_grid"] > 0.0) or not np.all(arrays["P_grid"] > 0.0):
            raise ValueError("T_grid and P_grid must be positive")
        if np.any(np.diff(arrays["T_grid"]) <= 0.0) or np.any(
            np.diff(arrays["P_grid"]) <= 0.0
        ):
            raise ValueError("T_grid and P_grid must be strictly increasing")

        n_bands = log_kggrid_np.shape[3]
        if (
            arrays["nu_bands"].ndim != 1
            or arrays["band_edges"].ndim != 2
            or arrays["nu_bands"].shape[0] != n_bands
            or arrays["band_edges"].shape != (n_bands, 2)
        ):
            raise ValueError(
                "Spectral band metadata does not match log_kggrid dimensions"
            )
        if not np.all(np.isfinite(arrays["nu_bands"])) or not np.all(
            np.isfinite(arrays["band_edges"])
        ):
            raise ValueError("Spectral band metadata must contain finite values")
        if not np.all(arrays["nu_bands"] > 0.0) or not np.all(
            arrays["band_edges"] > 0.0
        ):
            raise ValueError("Spectral band metadata must be positive")
        band_widths = arrays["band_edges"][:, 1] - arrays["band_edges"][:, 0]
        if not np.all(band_widths > 0.0):
            raise ValueError("Spectral band edges must have positive widths")
        if np.any(np.diff(arrays["nu_bands"]) <= 0.0):
            raise ValueError("Spectral band centers must be strictly increasing")
        if not np.all(
            (arrays["band_edges"][:, 0] <= arrays["nu_bands"])
            & (arrays["nu_bands"] <= arrays["band_edges"][:, 1])
        ):
            raise ValueError("Spectral band centers must lie within band edges")
        if np.any(arrays["band_edges"][1:, 0] < arrays["band_edges"][:-1, 1]):
            raise ValueError("Spectral band edges must not overlap")

        if arrays["band_edges"].size:
            inferred_band_width = float(
                arrays["band_edges"][0, 1] - arrays["band_edges"][0, 0]
            )
        else:
            inferred_band_width = None
        if "band_width" in meta:
            band_width = float(meta["band_width"])
        elif inferred_band_width is not None:
            band_width = inferred_band_width
        else:
            raise ValueError(
                "Missing band_width in metadata and cannot infer from band edges"
            )
        if not np.isfinite(band_width) or band_width <= 0.0:
            raise ValueError("band_width metadata must be finite and positive")
        band_spacing = str(meta.get("band_spacing", "log"))

        return dict(
            base_opa=base_opa,
            Ng=Ng_meta,
            band_width=band_width,
            band_spacing=band_spacing,
            arrays=arrays,
            expected_base_hash=expected_hash,
            expected_base_fingerprint=expected_meta,
        )

    def _apply_loaded_tables(self, payload):
        arrays = payload["arrays"]
        self.base_opa = payload["base_opa"]
        self.Ng = payload["Ng"]
        self.band_width = payload["band_width"]
        self.band_spacing = payload["band_spacing"]
        self.ckd_info = CKDTableInfo(
            log_kggrid=jnp.asarray(arrays["log_kggrid"]),
            ggrid=jnp.asarray(arrays["ggrid"]),
            weights=jnp.asarray(arrays["weights"]),
            T_grid=jnp.asarray(arrays["T_grid"]),
            P_grid=jnp.asarray(arrays["P_grid"]),
            nu_bands=jnp.asarray(arrays["nu_bands"]),
            band_edges=jnp.asarray(arrays["band_edges"]),
        )
        self.nu_bands = self.ckd_info.nu_bands
        self.band_edges = self.ckd_info.band_edges
        self.ready = True
        self._expected_base_meta = payload.get("expected_base_fingerprint")

    def load_tables(self, path: str, *, io_format: str = "npz", base_opa=None):
        effective_base_opa = base_opa if base_opa is not None else self.base_opa
        payload = self._load_tables_payload(effective_base_opa, path, io_format)
        self._apply_loaded_tables(payload)
        self._expected_base_hash = payload.get("expected_base_hash")
        return self

    @staticmethod
    def _infer_band_edges_from_centers(nu_centers: np.ndarray) -> np.ndarray:
        """Infer contiguous band edges from monotonic band centers."""
        nu_centers = np.asarray(nu_centers, dtype=float)
        if nu_centers.ndim != 1:
            raise ValueError("nu_centers must be a one-dimensional array")
        if nu_centers.size < 2:
            raise ValueError("At least two band centers are required to infer band edges")

        diffs = np.diff(nu_centers)
        if np.any(diffs == 0.0):
            raise ValueError("Band centers must be unique to infer band edges")
        if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
            raise ValueError("Band centers must be monotonic to infer band edges")

        midpoints = 0.5 * (nu_centers[:-1] + nu_centers[1:])
        edges = np.empty((nu_centers.size, 2), dtype=nu_centers.dtype)
        edges[0, 0] = nu_centers[0] - 0.5 * diffs[0]
        edges[0, 1] = midpoints[0]
        edges[1:-1, 0] = midpoints[:-1]
        edges[1:-1, 1] = midpoints[1:]
        edges[-1, 0] = midpoints[-1]
        edges[-1, 1] = nu_centers[-1] + 0.5 * diffs[-1]

        return np.sort(edges, axis=1)

    @staticmethod
    def _validate_external_table(
        xsgrid,
        samples,
        weights,
        temperatures,
        pressures,
        nu_centers,
    ) -> None:
        """Validate external CKD table arrays before constructing CKDTableInfo."""
        arrays = {
            "samples": np.asarray(samples),
            "weights": np.asarray(weights),
            "temperatures": np.asarray(temperatures),
            "pressures": np.asarray(pressures),
            "nu_centers": np.asarray(nu_centers),
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"External CKD {name} must be one-dimensional")
            if array.size == 0:
                raise ValueError(f"External CKD {name} must not be empty")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"External CKD {name} must contain finite values")

        xsgrid = np.asarray(xsgrid)
        if xsgrid.ndim != 4:
            raise ValueError("External CKD xsgrid must be four-dimensional")
        expected_shape = (
            arrays["temperatures"].size,
            arrays["pressures"].size,
            arrays["samples"].size,
            arrays["nu_centers"].size,
        )
        if xsgrid.shape != expected_shape:
            raise ValueError(
                "External CKD xsgrid shape does not match table axes: "
                f"expected {expected_shape}, got {xsgrid.shape}"
            )
        if not np.all((arrays["samples"] >= 0.0) & (arrays["samples"] <= 1.0)):
            raise ValueError("External CKD samples must lie within [0, 1]")
        if np.any(np.diff(np.sort(arrays["samples"])) <= 0.0):
            raise ValueError("External CKD samples must be unique")
        if arrays["weights"].size != arrays["samples"].size:
            raise ValueError("External CKD weights must match samples")
        if not np.all(arrays["weights"] > 0.0):
            raise ValueError("External CKD weights must be positive")
        if not np.isclose(np.sum(arrays["weights"]), 1.0, rtol=1.0e-5, atol=1.0e-8):
            raise ValueError("External CKD weights must sum to one")
        if not np.all(arrays["temperatures"] > 0.0):
            raise ValueError("External CKD temperatures must be positive")
        if not np.all(arrays["pressures"] > 0.0):
            raise ValueError("External CKD pressures must be positive")
        if not np.all(arrays["nu_centers"] > 0.0):
            raise ValueError("External CKD nu_centers must be positive")
        if np.unique(arrays["nu_centers"]).size != arrays["nu_centers"].size:
            raise ValueError("External CKD nu_centers must be unique")
        if np.any(np.diff(np.sort(arrays["temperatures"])) <= 0.0):
            raise ValueError("External CKD temperatures must be unique")
        if np.any(np.diff(np.sort(arrays["pressures"])) <= 0.0):
            raise ValueError("External CKD pressures must be unique")
        if not np.all(np.isfinite(xsgrid)):
            raise ValueError("External CKD xsgrid must contain finite values")
        if not np.all(xsgrid > 0.0):
            raise ValueError("External CKD xsgrid must be positive")

    def attach_base(self, base_opa, *, strict: bool = True) -> None:
        """attach base opacity calculator after loading tables."""
        actual = _hash_json(_base_fingerprint(base_opa))
        if strict and getattr(self, "_expected_base_hash", None) not in (None, actual):
            raise ValueError("base_opa fingerprint mismatch with loaded CKD table.")
        self.base_opa = base_opa

    @classmethod
    def from_saved_tables(cls, *args, io_format: str = "npz", base_opa=None, **kwargs):
        """Instantiate ``OpaCKD`` from a saved table.

        Supports both ``OpaCKD.from_saved_tables(path, base_opa=...)`` and the legacy
        calling pattern ``OpaCKD.from_saved_tables(base_opa, path)`` used in earlier
        code and tests.
        """
        if kwargs:
            raise TypeError(
                "from_saved_tables received unexpected keyword arguments: "
                f"{', '.join(kwargs)}"
            )

        if len(args) == 1:
            (path,) = args
        elif len(args) == 2:
            if base_opa is not None:
                raise TypeError(
                    "from_saved_tables received duplicate base_opa arguments"
                )
            base_opa, path = args
        else:
            raise TypeError(
                "from_saved_tables expects `(path)` or `(base_opa, path)` positional"
                " arguments"
            )

        inst = cls.load_only()
        return inst.load_tables(path, io_format=io_format, base_opa=base_opa)

    @classmethod
    def from_external(
        cls, provider: str, path: str, nurange: Optional[Sequence[float]] = None
    ):
        """Instantiate ``OpaCKD`` from an external CKD table provider.

        Args:
            provider: Name of the CKD table provider, such as ``"exomolop"``.
            path: Path to the CKD table file or directory
            nurange: Optional ``(nu_min, nu_max)`` wavenumber window. If given, only
                bands whose inferred edges overlap the range are loaded.

        Currently supports provider ``\"exomolop\"`` which follows the return contract
        of :func:`exojax.provider.exomolop.load_ckd`.
        """
        provider_key = provider.lower()
        if provider_key != "exomolop":
            raise ValueError(f"Unsupported CKD provider '{provider}'.")

        from exojax.provider import exomolop as exomolop_provider
        from exojax.provider.exomolop import download_exomolop_h5

        # check path is file or directory
        import pathlib
        path = pathlib.Path(path).expanduser()
        if path.is_dir():
            h5_paths = sorted(path.glob("*.h5"))
            nonempty_h5_paths = [
                h5_path for h5_path in h5_paths if h5_path.stat().st_size > 0
            ]
            if len(nonempty_h5_paths) == 1:
                path = nonempty_h5_paths[0]
            elif len(nonempty_h5_paths) > 1:
                raise ValueError(
                    f"Multiple non-empty CKD h5 files found in {path}. "
                    "Specify the file path."
                )
            else:
                path = download_exomolop_h5(path)
        elif not path.suffix:
            # download ExoMol opacity file
            path = download_exomolop_h5(path)
        elif path.suffix != ".h5":
            raise ValueError(f"CKD table file must have .h5 suffix: {path}")
        elif not path.exists():
            raise FileNotFoundError(f"CKD table file does not exist: {path}")
        elif path.stat().st_size == 0:
            raise ValueError(f"CKD table file is empty: {path}")

        (
            xsgrid,
            samples,
            weights,
            temperatures,
            pressures,
            nu_centers,
            _molecule,
            molmass,
        ) = exomolop_provider.load_ckd(path)

        molmass = np.asarray(molmass, dtype=float)
        if molmass.size != 1:
            raise ValueError("External CKD molmass must be scalar")
        molmass = float(molmass.reshape(-1)[0])
        if not np.isfinite(molmass) or molmass <= 0.0:
            raise ValueError("External CKD molmass must be finite and positive")
        cls._validate_external_table(
            xsgrid, samples, weights, temperatures, pressures, nu_centers
        )
        temperature_order = np.argsort(temperatures)
        pressure_order = np.argsort(pressures)
        sample_order = np.argsort(samples)
        temperatures = temperatures[temperature_order]
        pressures = pressures[pressure_order]
        samples = samples[sample_order]
        weights = weights[sample_order]
        xsgrid = xsgrid[temperature_order][:, pressure_order, :, :]
        xsgrid = xsgrid[:, :, sample_order, :]
        band_order = np.argsort(nu_centers)
        nu_centers = nu_centers[band_order]
        xsgrid = xsgrid[..., band_order]
        band_edges = cls._infer_band_edges_from_centers(nu_centers)

        if nurange is not None:
            nurange = np.asarray(nurange)
            if nurange.ndim != 1 or nurange.size < 2:
                raise ValueError(
                    "nurange must be a 2 or more -element sequence "
                    "(nu_min, ..., nu_max)"
                )
            if not np.all(np.isfinite(nurange)):
                raise ValueError("nurange must contain finite values")
            if not np.all(nurange > 0.0):
                raise ValueError("nurange must contain positive wavenumbers")
            nu_min = nurange[0]
            nu_max = nurange[-1]
            if nu_min > nu_max:
                raise ValueError("nurange must satisfy nu_min <= nu_max")
            nu_mask = (band_edges[:, 1] >= nu_min) & (band_edges[:, 0] <= nu_max)
            if not np.any(nu_mask):
                raise ValueError(
                    "Requested nurange does not overlap any CKD wavenumber bands"
                )
            xsgrid = xsgrid[..., nu_mask]
            nu_centers = nu_centers[nu_mask]
            band_edges = band_edges[nu_mask]

        inst = cls.load_only()
        inst.Ng = int(len(samples))
        inst.band_width = float(np.median(band_edges[:, 1] - band_edges[:, 0]))
        inst.band_spacing = "external"
        inst.ckd_info = CKDTableInfo(
            log_kggrid=jnp.log(jnp.asarray(xsgrid)),
            ggrid=jnp.asarray(samples),
            weights=jnp.asarray(weights),
            T_grid=jnp.asarray(temperatures),
            P_grid=jnp.asarray(pressures),
            nu_bands=jnp.asarray(nu_centers),
            band_edges=jnp.asarray(band_edges),
        )
        inst.nu_bands = inst.ckd_info.nu_bands
        inst.band_edges = inst.ckd_info.band_edges
        inst.molmass = molmass
        inst.ready = True
        return inst
