"""Default provider implementations for OpaPremodit behavior injection.

These keep only NumPy arrays and lightweight state, avoiding heavy dependencies.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap

from exojax.opacity.contracts import PartitionFunctionProvider, BroadeningStrategy
from exojax.database._common.partition_function import qr_interp as qr_interp_hitran
from exojax.database.exomol.partition_function import qr_interp as qr_interp_exomol
from exojax.opacity.premodit.core import (
    _compute_broadening_parameters_hitran,
    _compute_broadening_parameters_exomol,
)


class ExomolPartitionProvider(PartitionFunctionProvider):
    def __init__(self, T_gQT: np.ndarray, gQT: np.ndarray):
        self.T_gQT = T_gQT
        self.gQT = gQT

    def qr_single(self, T: float, Tref: float):
        return qr_interp_exomol(T, Tref, self.T_gQT, self.gQT)

    def qr_vector(self, Tarr: np.ndarray, Tref: float):
        Tarr = jnp.asarray(Tarr)
        fn = lambda T: qr_interp_exomol(T, Tref, self.T_gQT, self.gQT)
        return vmap(fn, in_axes=0)(Tarr)


class HitranPartitionProvider(PartitionFunctionProvider):
    def __init__(self, isotope, uniqiso, T_gQT: np.ndarray, gQT: np.ndarray):
        self.isotope = isotope
        self.uniqiso = uniqiso
        self.T_gQT = T_gQT
        self.gQT = gQT

    def qr_single(self, T: float, Tref: float):
        return qr_interp_hitran(self.isotope, self.uniqiso, T, Tref, self.T_gQT, self.gQT)

    def qr_vector(self, Tarr: np.ndarray, Tref: float):
        Tarr = jnp.asarray(Tarr)
        fn = lambda T: qr_interp_hitran(self.isotope, self.uniqiso, T, Tref, self.T_gQT, self.gQT)
        return vmap(fn, in_axes=0)(Tarr)


class ExomolBroadening(BroadeningStrategy):
    def __init__(self, n_Texp: np.ndarray, alpha_ref: np.ndarray):
        self._n_Texp = n_Texp
        self._alpha_ref = alpha_ref

    def compute(self, Tref_broadening: float) -> Tuple[np.ndarray, np.ndarray]:
        return _compute_broadening_parameters_exomol(self._n_Texp, self._alpha_ref, Tref_broadening)


class HitranBroadening(BroadeningStrategy):
    def __init__(self, n_air: np.ndarray, gamma_air: np.ndarray):
        self._n_air = n_air
        self._gamma_air = gamma_air

    def compute(self, Tref_broadening: float) -> Tuple[np.ndarray, np.ndarray]:
        return _compute_broadening_parameters_hitran(self._n_air, self._gamma_air, Tref_broadening)
