from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


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

