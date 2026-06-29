from dataclasses import dataclass
from typing import Any, Optional, Literal, Protocol

import numpy as np

ArrayLike = Any


@dataclass(frozen=True)
class MDBMeta:
    """Immutable metadata required across opacity calculators."""

    dbtype: Literal["hitran", "exomol"]
    molmass: float
    T_gQT: np.ndarray
    gQT: np.ndarray


@dataclass(frozen=True)
class Lines:
    """Selected, already-filtered line payload used in opacity calculations."""

    nu_lines: np.ndarray
    elower: np.ndarray
    line_strength_ref_original: np.ndarray


@dataclass(frozen=True)
class MDBSnapshot:
    """Data-only snapshot combining metadata and line/broadening payloads.

    Notes:
        - ExoMol-only fields: ``n_Texp``, ``alpha_ref``.
        - HITRAN-only fields: ``isotope``, ``uniqiso``, ``n_air``, ``gamma_air``.
    """

    meta: MDBMeta
    lines: Lines

    # ExoMol-only
    n_Texp: Optional[np.ndarray] = None
    alpha_ref: Optional[np.ndarray] = None

    # HITRAN-only
    isotope: Optional[np.ndarray] = None
    uniqiso: Optional[np.ndarray] = None
    n_air: Optional[np.ndarray] = None
    gamma_air: Optional[np.ndarray] = None


class DirectLineDatabase(Protocol):
    """Contract for line databases used by direct Voigt opacity calculators."""

    dbtype: str
    Tref: float

    nu_lines: ArrayLike
    logsij0: ArrayLike
    A: ArrayLike
    elower: ArrayLike
    line_masses: ArrayLike

    def generate_jnp_arrays(self) -> None:
        """Generate JAX arrays."""
        ...

    def qr_interp_lines(self, T: float, Tref: float) -> ArrayLike:
        """Interpolate partition-function ratios for selected lines.

        Args:
            T: Temperature in K.
            Tref: Reference temperature in K.

        Returns:
            Partition-function ratios.
        """
        ...
