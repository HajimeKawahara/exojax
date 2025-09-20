"""Lightweight provider contracts for opacity behavior injection.

Defines Protocols for partition function interpolation and broadening
parameter computation so OpaPremodit can avoid dbtype-specific branches.
"""

from __future__ import annotations

from typing import Protocol, Tuple
import numpy as np


class PartitionFunctionProvider(Protocol):
    """Provides Q(T)/Q(Tref) for single T or a vector of T."""

    def qr_single(self, T: float, Tref: float) -> np.ndarray:
        """Return partition function ratio for a single temperature."""

    def qr_vector(self, Tarr: np.ndarray, Tref: float) -> np.ndarray:
        """Return partition function ratios for a vector of temperatures."""


class BroadeningStrategy(Protocol):
    """Computes (n_Texp, gamma_ref) at the given reference temperature."""

    def compute(self, Tref_broadening: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (n_Texp, gamma_ref) arrays at Tref_broadening."""

