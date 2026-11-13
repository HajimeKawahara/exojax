"""Hydrogen atomic database."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database.core.line_strength import line_strength
from exojax.database.core_atom._arrays import (
    _LINE_FIELDS,
    _generate_atomic_jnp_arrays,
    _mask_atomic_lines,
    _set_atomic_metadata,
)
from exojax.database.core_atom.io import load_pf_Barklem2016
from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.pf import interp_QT_284
from exojax.database.core_atom.transition import (
    einstein_a_from_oscillator_strength,
)
from exojax.utils.constants import Tref_original

__all__ = ["AdbHydrogen"]

_BALMER_F_LU = {
    3: 0.6407,
    4: 0.1193,
    5: 0.0446,
    6: 0.0221,
    7: 0.0127,
    8: 0.00803,
    9: 0.00543,
    10: 0.00384,
}

_HYDROGEN_RYDBERG_CM = 109677.5834
_HYDROGEN_LINE_FIELDS = _LINE_FIELDS + ("glower",)


class AdbHydrogen:
    """Atomic database for hydrogen lines."""

    def __init__(
        self,
        nurange=(-np.inf, np.inf),
        margin=0.0,
        crit=0.0,
        series="balmer",
        n_upper_min=3,
        n_upper_max=10,
        gpu_transfer=True,
        vmr_fraction=None,
    ):
        """Initialize hydrogen atomic lines.

        Args:
            nurange: Wavenumber range in cm-1.
            margin: Margin added to the wavenumber range in cm-1.
            crit: Reference line-strength lower cutoff.
            series: Hydrogen series name. Only ``"balmer"`` is supported.
            n_upper_min: Minimum upper principal quantum number.
            n_upper_max: Maximum upper principal quantum number.
            gpu_transfer: If True, create JAX arrays.
            vmr_fraction: VMR fractions of H, He, and H2.
        """
        if series != "balmer":
            raise NotImplementedError(
                "AdbHydrogen currently supports only Balmer lines."
            )

        self.dbtype = "hydrogen"
        self.series = series
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit
        if vmr_fraction is None:
            self.vmrH, self.vmrHe, self.vmrHH = [0.0, 0.16, 0.84]
        else:
            self.vmrH, self.vmrHe, self.vmrHH = vmr_fraction

        upper = np.arange(n_upper_min, n_upper_max + 1, dtype=int)
        missing = [n for n in upper if n not in _BALMER_F_LU]
        if missing:
            raise ValueError(f"Missing Balmer oscillator strengths for n={missing}.")

        lower = np.full_like(upper, 2)
        self.n_lower = lower
        self.n_upper = upper
        self.nu_lines = _HYDROGEN_RYDBERG_CM * (1.0 / lower**2 - 1.0 / upper**2)
        self._elower = _hydrogen_level_energy_cm(lower)
        self._eupper = _hydrogen_level_energy_cm(upper)
        self._glower = _hydrogen_level_degeneracy(lower)
        self._gupper = _hydrogen_level_degeneracy(upper)
        self._jlower = np.zeros_like(self.nu_lines)
        self._jupper = np.zeros_like(self.nu_lines)
        self._ielem = np.ones_like(upper)
        self._iion = np.ones_like(upper)

        f_lu = np.array([_BALMER_F_LU[int(n)] for n in upper])
        self._A = einstein_a_from_oscillator_strength(
            self.nu_lines,
            f_lu,
            self._glower,
            self._gupper,
        )
        self._gamRad = np.log10(self._A)
        self._gamSta = np.zeros_like(self.nu_lines)
        self._vdWdamp = np.full_like(self.nu_lines, -99.0)

        pfTdat, self.pfdat = load_pf_Barklem2016()
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(dtype=float))
        self.Tref = Tref_original
        self.QTref_284 = np.array(
            interp_QT_284(Tref_original, self.T_gQT, self.gQT_284species)
        )
        self._QTmask = self.make_QTmask()
        self.Sij0 = line_strength_atom(
            self._A,
            self._gupper,
            self.nu_lines,
            self._elower,
            self.QTref_284,
            self._QTmask,
        )

        mask = (
            (self.nu_lines > self.nurange[0] - self.margin)
            * (self.nu_lines < self.nurange[1] + self.margin)
            * (self.Sij0 > self.crit)
        )
        self.masking(mask)
        _set_atomic_metadata(self)
        if gpu_transfer:
            self.generate_jnp_arrays()

    @property
    def line_masses(self):
        """Atomic masses for the current line selection, in amu."""
        return self.atomicmass

    def make_QTmask(self):
        """Return the H I partition-function index.

        Returns:
            Index array for the H I partition function.
        """
        qtmask = np.where(self.pfdat["T[K]"] == "H_I")[0][0]
        return np.full_like(self.nu_lines, qtmask, dtype=int)

    def masking(self, mask):
        """Select lines and metadata, refreshing existing JAX arrays.

        Select lines before constructing an OpaDirect instance.

        Args:
            mask: Boolean mask for selected lines.
        """
        self.n_lower = self.n_lower[mask]
        self.n_upper = self.n_upper[mask]
        _mask_atomic_lines(self, mask, line_fields=_HYDROGEN_LINE_FIELDS)
        if len(self.nu_lines) < 1:
            warnings.warn("No hydrogen lines are selected.", UserWarning)

    def generate_jnp_arrays(self):
        """Generate JAX arrays for the current line selection."""
        _generate_atomic_jnp_arrays(self, line_fields=_HYDROGEN_LINE_FIELDS)

    def Atomic_gQT(self, atomspecies="H 1"):
        """Return the H I partition-function grid.

        Args:
            atomspecies: Atomic species label.

        Returns:
            Partition-function grid.
        """
        if atomspecies != "H 1":
            raise ValueError("AdbHydrogen contains only H 1.")
        return self.gQT_284species[np.where(self.pfdat["T[K]"] == "H_I")][0]

    def QT_interp(self, atomspecies, T):
        """Interpolate the H I partition function.

        Args:
            atomspecies: Atomic species label.
            T: Temperature in K.

        Returns:
            Partition function.
        """
        return jnp.interp(T, self.T_gQT, self.Atomic_gQT(atomspecies))

    def qr_interp(self, atomspecies, T):
        """Interpolate the H I partition-function ratio.

        Args:
            atomspecies: Atomic species label.
            T: Temperature in K.

        Returns:
            Partition-function ratio Q(T)/Q(Tref).
        """
        qt = self.QT_interp(atomspecies, T)
        qtref = self.QTref_284[self.make_QTmask()[0]]
        return qt / qtref

    def qr_interp_lines(self, T, Tref):
        """Interpolate H I partition-function ratios for selected lines.

        Args:
            T: Temperature in K.
            Tref: Reference temperature in K.

        Returns:
            Partition-function ratios for each line.
        """
        qt = self.QT_interp("H 1", T)
        qtref = self.QT_interp("H 1", Tref)
        return jnp.ones_like(self.logsij0) * qt / qtref

    def line_strength(self, T):
        """Compute line strengths at temperature.

        Args:
            T: Temperature in K.

        Returns:
            Line strengths in cm.
        """
        qr = self.qr_interp_lines(T, self.Tref)
        return line_strength(T, self.logsij0, self.nu_lines, self.elower, qr, self.Tref)


def _hydrogen_level_energy_cm(n):
    return _HYDROGEN_RYDBERG_CM * (1.0 - 1.0 / np.asarray(n, dtype=float) ** 2)


def _hydrogen_level_degeneracy(n):
    return 2.0 * np.asarray(n, dtype=float) ** 2

