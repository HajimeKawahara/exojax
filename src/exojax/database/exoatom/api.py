"""ExoAtom atomic lines read with PyExoCross, for direct Voigt opacity."""

import json
from pathlib import Path
import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database.core.line_strength import line_strength
from exojax.database.core_atom._arrays import (
    _generate_atomic_jnp_arrays, _mask_atomic_lines,
)
from exojax.database.core_atom.io import PeriodicTable
from exojax.database.exoatom._files import dataset_identity, ensure_exoatom_files
from exojax.utils.constants import Tref_original, ccgs, hcperk

__all__ = ["AdbExoAtom"]

_LINE_FIELDS = (
    "A", "elower", "eupper", "glower", "gupper", "jlower", "jupper",
    "ielem", "iion", "i_upper", "i_lower", "logsij0", "gamma_natural",
)


class AdbExoAtom:
    """One ExoAtom species, with its source mass and partition function.

    ``OpaDirect`` uses natural widths from both level lifetimes when available.
    Otherwise supply an explicit ``atomic_broadening(T, P)`` callback returning
    total Lorentzian HWHM in cm-1. No pressure broadening is assumed.
    """

    def __init__(
        self, path, nurange=(-np.inf, np.inf), *, margin=0.0, crit=0.0,
        elower_max=None, gpu_transfer=True, local_databases="./",
        download=True, Tref=Tref_original,
    ):
        """Read raw ExoAtom data through the optional PyExoCross dependency.

        Args:
            path: Dataset directory, e.g. ``Li/NIST``, ``Li_p/NIST``,
                ``Li/Kurucz``, or ``H/1H/NIST``.
            nurange: Wavenumber bounds or grid in cm-1.
            margin: Nonnegative additional coverage on both sides, in cm-1.
            crit: Nonnegative line-strength cutoff at Tref, in cm.
                Zero keeps weak lines even if their linear strengths underflow.
            elower_max: Optional upper bound on lower energy in cm-1.
            gpu_transfer: Generate JAX arrays; if False, call
                ``generate_jnp_arrays()`` before constructing OpaDirect.
            local_databases: Root directory for relative dataset paths.
            download: Fetch missing files if True; False requires local files.
            Tref: Reference temperature in K, within the source partition table.

        Notes:
            The reader loads states into memory and streams transitions without
            creating parsed-file caches. It retains supplied line positions,
            fractional J and source statistical weights. The source .pf table
            is used without substituting a Barklem or Irwin partition function.
        """
        self.path = (Path(local_databases).expanduser() / Path(path).expanduser()).resolve()
        atom, charge, _ = dataset_identity(self.path)
        bounds = np.asarray(nurange, dtype=float)
        if bounds.ndim != 1 or bounds.size < 2 or np.isnan(bounds).any():
            raise ValueError("nurange must contain at least two wavenumbers without NaN.")
        self.nurange = [float(np.min(bounds)), float(np.max(bounds))]
        if self.nurange[0] >= self.nurange[1] or self.nurange[1] <= 0:
            raise ValueError("nurange must span a positive wavenumber interval.")
        if not np.isfinite(margin) or margin < 0 or not np.isfinite(crit) or crit < 0:
            raise ValueError("margin and crit must be finite and nonnegative.")
        if elower_max is not None and (not np.isfinite(elower_max) or elower_max < 0):
            raise ValueError("elower_max must be finite and nonnegative.")
        if not np.isfinite(Tref) or Tref <= 0:
            raise ValueError("Tref must be finite and positive.")

        from exojax.database._common.pyexocross import import_pyexocross, read_exomol_lines

        import_pyexocross()
        from pyexocross.base.qn_metadata import normalized_states_columns

        self.dbtype, self.backend = "exoatom", "pyexocross"
        self.Tref, self.margin, self.crit = float(Tref), margin, crit
        self.elower_max, self.gpu_transfer = elower_max, gpu_transfer
        files = ensure_exoatom_files(self.path, download=download)
        definition = json.loads(files["adef.json"].read_text())
        species, dataset = definition["species"], definition["dataset"]
        if species["atom"] != atom or species["charge"] != charge:
            raise ValueError("ExoAtom species metadata do not match the requested path.")
        if dataset["name"] != self.path.name:
            raise ValueError("ExoAtom dataset metadata do not match the requested path.")
        if dataset["transitions"]["number_of_transition_files"] != 1:
            raise NotImplementedError("ExoAtom currently supports one transition file per dataset.")
        self.species = atom + ("_II" if charge else "_I")
        self.database = dataset["name"]
        self.mass = float(definition.get("isotope", {}).get("mass", species["mass_in_Da"]))
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("ExoAtom mass_in_Da must be finite and positive.")
        partition = np.loadtxt(files["pf"], usecols=(0, 1), ndmin=2)
        if (len(partition) < 2 or not np.isfinite(partition).all()
                or np.any(partition <= 0) or np.any(np.diff(partition[:, 0]) <= 0)):
            raise ValueError("ExoAtom partition temperatures must increase and values must be positive.")
        self.T_gQT, self.gQT = partition.T
        if not self.T_gQT[0] <= Tref <= self.T_gQT[-1]:
            raise ValueError("Tref must lie within the ExoAtom partition-function table.")
        states = dataset["states"]
        columns = normalized_states_columns([field["name"] for field in states["states_file_fields"]])
        # Optional g-factor/QN descriptions can disagree with actual NIST files.
        # Only numeric fields needed by direct opacity are requested.
        lifetime = states.get("lifetime_available", False)
        if lifetime and "tau" not in columns:
            raise ValueError("ExoAtom lifetime metadata are missing the tau column.")
        expanded = [self.nurange[0] - margin, self.nurange[1] + margin]
        frame = read_exomol_lines(
            files["states"], [files["trans"]], columns, expanded,
            extra_state_columns=["tau"] if lifetime else [],
        ).sort_values("nu_lines", kind="stable")
        required = ["nu_lines", "A", "elower", "eupper", "glower", "gup"]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise ValueError("ExoAtom lines contain non-finite required parameters.")
        # Some atomic levels have no resolved J. Preserve NaN rather than
        # infer a quantum number from a summed statistical weight.
        if (np.any(frame[["nu_lines", "A", "glower", "gup"]].to_numpy() <= 0)
                or np.any(frame[["elower", "eupper", "jlower", "jupper"]].to_numpy() < 0)
                or np.isinf(frame[["jlower", "jupper"]].to_numpy()).any()):
            raise ValueError("ExoAtom lines contain invalid energies, weights, J, or transition values.")
        self.nu_lines = frame.nu_lines.to_numpy()
        for name in ("A", "elower", "eupper", "glower", "jlower", "jupper", "i_upper", "i_lower"):
            setattr(self, "_" + name, frame[name].to_numpy())
        self._gupper = frame.gup.to_numpy()
        nline = len(frame)
        self._ielem = np.full(nline, np.flatnonzero(PeriodicTable == atom)[0], dtype=int)
        self._iion = np.full(nline, charge + 1, dtype=int)
        self.atomicmass = np.full(nline, self.mass)
        qref = np.interp(Tref, self.T_gQT, self.gQT)
        self._logsij0 = (
            np.log(self._A) + np.log(self._gupper) - np.log(8.0 * np.pi * ccgs)
            - 2.0 * np.log(self.nu_lines) - np.log(qref)
            - hcperk * self._elower / Tref
            + np.log(-np.expm1(-hcperk * self.nu_lines / Tref))
        )
        self.Sij0 = np.exp(self._logsij0)
        self._gamma_natural = np.full(nline, np.nan)
        if lifetime:
            lower, upper = frame.tau_l.to_numpy(dtype=float), frame.tau_u.to_numpy(dtype=float)
            valid = (lower > 0) & (upper > 0)
            self._gamma_natural[valid] = (
                1.0 / lower[valid] + 1.0 / upper[valid]
            ) / (4.0 * np.pi * ccgs)
        mask = np.isfinite(self._logsij0)
        if crit > 0:
            mask &= self._logsij0 > np.log(crit)
        if elower_max is not None:
            mask &= self._elower < elower_max
        self.local_paths = [str(file) for file in files.values()]
        self.masking(mask)
        if gpu_transfer:
            self.generate_jnp_arrays()

    def masking(self, mask):
        """Select current lines, preserving host and existing device alignment."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != self.nu_lines.shape:
            raise ValueError("mask must be a Boolean array with one entry per current line.")
        _mask_atomic_lines(self, mask, line_fields=_LINE_FIELDS, metadata_fields=("atomicmass",))
        if not self.nu_lines.size:
            warnings.warn("No ExoAtom lines are selected.", UserWarning, stacklevel=2)

    apply_mask_mdb = masking

    def generate_jnp_arrays(self):
        """Generate JAX arrays, retaining finite logarithms of weak lines."""
        _generate_atomic_jnp_arrays(self, line_fields=_LINE_FIELDS, metadata_fields=("atomicmass",))

    @property
    def line_masses(self):
        """Mass in amu for each selected line."""
        return self.atomicmass

    def QT_interp(self, T):
        """Interpolate the source partition table; return NaN outside its range."""
        return jnp.interp(T, self.T_gQT, self.gQT, left=jnp.nan, right=jnp.nan)

    def qr_interp_lines(self, T, Tref):
        """Partition-function ratios for the current line selection."""
        return jnp.full(self.nu_lines.shape, self.QT_interp(T) / self.QT_interp(Tref))

    def line_strength(self, T):
        """Line strengths in cm at temperature T, including weak hot lines."""
        return line_strength(
            T, self._logsij0, self.nu_lines, self._elower,
            self.qr_interp_lines(T, self.Tref), self.Tref,
        )
