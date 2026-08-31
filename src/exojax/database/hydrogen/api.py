"""Hydrogen atomic database."""

import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database.core.line_strength import line_strength
from exojax.database.core_atom.io import load_atomicdata
from exojax.database.core_atom.io import load_ionization_energies
from exojax.database.core_atom.io import load_pf_Barklem2016
from exojax.database.core_atom.io import pick_ionE
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
        self._gupper = _hydrogen_level_degeneracy(upper)
        self._jlower = np.zeros_like(self.nu_lines)
        self._jupper = np.zeros_like(self.nu_lines)
        self._ielem = np.ones_like(upper)
        self._iion = np.ones_like(upper)

        f_lu = np.array([_BALMER_F_LU[int(n)] for n in upper])
        self._A = einstein_a_from_oscillator_strength(
            self.nu_lines,
            f_lu,
            _hydrogen_level_degeneracy(lower),
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
        if gpu_transfer:
            self.generate_jnp_arrays()

        ipccd = load_atomicdata()
        self.solarA = jnp.array(
            [ipccd[ipccd["ielem"] == 1].iat[0, 4]] * len(self.nu_lines)
        )
        self.atomicmass = jnp.array(
            [ipccd[ipccd["ielem"] == 1].iat[0, 5]] * len(self.nu_lines)
        )
        self.line_masses = self.atomicmass
        df_ionE = load_ionization_energies()
        self.ionE = jnp.array([pick_ionE(1, 1, df_ionE)] * len(self.nu_lines))

    def make_QTmask(self):
        """Return the H I partition-function index.

        Returns:
            Index array for the H I partition function.
        """
        qtmask = np.where(self.pfdat["T[K]"] == "H_I")[0][0]
        return np.full_like(self.nu_lines, qtmask, dtype=int)

    def masking(self, mask):
        """Apply a line mask.

        Args:
            mask: Boolean mask for selected lines.
        """
        self.n_lower = self.n_lower[mask]
        self.n_upper = self.n_upper[mask]
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]
        self._A = self._A[mask]
        self._elower = self._elower[mask]
        self._eupper = self._eupper[mask]
        self._gupper = self._gupper[mask]
        self._jlower = self._jlower[mask]
        self._jupper = self._jupper[mask]
        self._QTmask = self._QTmask[mask]
        self._ielem = self._ielem[mask]
        self._iion = self._iion[mask]
        self._gamRad = self._gamRad[mask]
        self._gamSta = self._gamSta[mask]
        self._vdWdamp = self._vdWdamp[mask]
        if len(self.nu_lines) < 1:
            warnings.warn("No hydrogen lines are selected.", UserWarning)

    def generate_jnp_arrays(self):
        """Generate JAX arrays."""
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.Sij0))
        self.A = jnp.array(self._A)
        self.elower = jnp.array(self._elower)
        self.eupper = jnp.array(self._eupper)
        self.gupper = jnp.array(self._gupper)
        self.jlower = jnp.array(self._jlower, dtype=int)
        self.jupper = jnp.array(self._jupper, dtype=int)
        self.QTmask = jnp.array(self._QTmask, dtype=int)
        self.ielem = jnp.array(self._ielem, dtype=int)
        self.iion = jnp.array(self._iion, dtype=int)
        self.gamRad = jnp.array(self._gamRad)
        self.gamSta = jnp.array(self._gamSta)
        self.vdWdamp = jnp.array(self._vdWdamp)

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

