"""NIST atomic transitions fetched through the RADIS database API."""

import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database._common.radis_adapter import fetch_nist_lines
from exojax.database.core_atom._arrays import (
    _generate_atomic_jnp_arrays,
    _mask_atomic_lines,
)
from exojax.database.core_atom.io import load_atomicdata, load_pf_Barklem2016
from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.pf import interp_QT_284, qr_interp_lines
from exojax.database.nist._convert import line_arrays, parse_species
from exojax.utils.constants import Tref_original

__all__ = ["AdbNist"]

_LINE_FIELDS = (
    "A", "elower", "eupper", "glower", "gupper", "jlower", "jupper",
    "QTmask", "ielem", "iion",
)


class AdbNist:
    """NIST lines for one atomic species, without assumed damping parameters.

    NIST supplies transition probabilities, not a complete broadening model.
    Use ``OpaDirect(..., atomic_broadening=...)`` to specify total Lorentzian
    half widths. Partition functions use Barklem & Collet (2016), optionally
    replacing Fe I with Irwin (1981).
    """

    def __init__(
        self, species, nurange=(-np.inf, np.inf), *, local_databases=None,
        databank_name="ExoJAX-NIST-{molecule}",
        margin=0.0, crit=0.0, Irwin=False, gpu_transfer=True,
        engine="pytables", cache=True,
    ):
        """Load and select NIST atomic lines.

        Args:
            species: One species in ``'Fe_II'`` or ``'Fe II'`` notation.
            nurange: Wavenumber bounds or grid in cm-1.
            local_databases: RADIS cache directory; None uses its default.
            databank_name: RADIS registry name. Use a different name when
                storing the same species in another cache directory.
            margin: Nonnegative wavenumber margin in cm-1.
            crit: Nonnegative reference line-strength cutoff in cm.
            Irwin: Use Irwin (1981) for Fe I only.
            gpu_transfer: Generate JAX arrays at initialization when True.
            engine: RADIS cache engine, defaulting to PyTables.
            cache: Cache option passed to RADIS without modification.

        Raises:
            ValueError: If the species, partition function, or selection is invalid.
        """
        self.species, ielem, iion = parse_species(species)
        self.dbtype = "nist"
        self.Tref = Tref_original
        self.Irwin = Irwin
        self.gpu_transfer = gpu_transfer
        bounds = np.asarray(nurange, dtype=float)
        if bounds.ndim != 1 or bounds.size < 2 or np.any(np.isnan(bounds)):
            raise ValueError("nurange must contain at least two wavenumbers without NaN.")
        self.nurange = [float(np.min(bounds)), float(np.max(bounds))]
        if self.nurange[0] >= self.nurange[1] or self.nurange[1] <= 0:
            raise ValueError("nurange must span a positive wavenumber interval.")
        if not np.isfinite(margin) or margin < 0 or not np.isfinite(crit) or crit < 0:
            raise ValueError("margin and crit must be finite and nonnegative.")
        self.margin, self.crit = margin, crit
        self.engine = engine

        pf_temperature, self.pfdat = load_pf_Barklem2016()
        matches = np.flatnonzero(self.pfdat["T[K]"].to_numpy() == self.species)
        if matches.size != 1:
            raise ValueError(f"No Barklem partition function is available for '{self.species}'.")
        atomic_data = load_atomicdata().set_index("ielem")
        if ielem not in atomic_data.index:
            raise ValueError(f"No atomic mass is available for '{self.species}'.")
        mass = float(atomic_data.loc[ielem, "mass"])
        self.T_gQT = jnp.asarray(pf_temperature.columns[1:], dtype=float)
        self.gQT_284species = jnp.asarray(self.pfdat.iloc[:, 1:].to_numpy(dtype=float))
        self.QTref_284 = np.asarray(interp_QT_284(
            self.Tref, self.T_gQT, self.gQT_284species, self.Irwin
        ))

        expanded = [self.nurange[0] - margin, self.nurange[1] + margin]
        fetch_bounds = [value if np.isfinite(value) else None for value in expanded]
        frame, self.local_paths = fetch_nist_lines(
            self.species, nurange=fetch_bounds, local_databases=local_databases,
            databank_name=databank_name,
            engine=engine, cache=cache,
        )
        arrays = line_arrays(frame, self.species)
        self.nu_lines = arrays.pop("nu_lines")
        for name, values in arrays.items():
            setattr(self, "_" + name, values)
        nline = len(self.nu_lines)
        self._ielem = np.full(nline, ielem, dtype=int)
        self._iion = np.full(nline, iion, dtype=int)
        self._QTmask = np.full(nline, matches[0], dtype=int)
        self.atomicmass = np.full(nline, mass)
        self.Sij0 = line_strength_atom(
            self._A, self._gupper, self.nu_lines, self._elower,
            self.QTref_284, self._QTmask,
        )
        self.masking(
            (self.nu_lines > expanded[0]) & (self.nu_lines < expanded[1])
            & np.isfinite(self.Sij0) & (self.Sij0 > crit)
        )
        if gpu_transfer:
            self.generate_jnp_arrays()

    def masking(self, mask):
        """Select current lines and refresh any existing JAX arrays."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != self.nu_lines.shape:
            raise ValueError("mask must be a Boolean array with one entry per current line.")
        _mask_atomic_lines(
            self, mask, line_fields=_LINE_FIELDS, metadata_fields=("atomicmass",)
        )
        if self.nu_lines.size == 0:
            warnings.warn("No NIST lines are selected.", UserWarning, stacklevel=2)

    def generate_jnp_arrays(self):
        """Generate JAX arrays while retaining host line data."""
        _generate_atomic_jnp_arrays(
            self, line_fields=_LINE_FIELDS, metadata_fields=("atomicmass",)
        )

    @property
    def line_masses(self):
        """Atomic masses for the current line selection, in amu."""
        return self.atomicmass

    def qr_interp_lines(self, T, Tref):
        """Return the partition-function ratio for every selected line."""
        return qr_interp_lines(
            T, Tref, self.T_gQT, self.gQT_284species, self._QTmask, self.Irwin
        )
