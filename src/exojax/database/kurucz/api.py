"""Atomic database (MDB) class."""

import pathlib
import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database.core_atom._arrays import (
    _generate_atomic_jnp_arrays,
    _mask_atomic_lines,
    _set_atomic_metadata,
)
from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.pf import interp_QT_284
from exojax.database.core_atom.pf import partfn_Fe
from exojax.database.core_atom.pf import qr_interp_lines

from exojax.database.core_atom.io import read_kurucz
from exojax.database.core_atom.io import load_pf_Barklem2016
from exojax.database.core_atom.io import PeriodicTable
from exojax.database._common.radis_adapter import (
    fetch_kurucz_dataframe,
    get_radis_version,
)
from exojax.utils.constants import Tref_original

__all__ = ["AdbKurucz"]

explanation_states = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
explanation_trans = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
warning_old_exojax = "It seems that the hdf5 file for the transition file was created using the old version of exojax<1.1. Try again after removing "


class AdbKurucz:
    """atomic database from Kurucz (http://kurucz.harvard.edu/linelists/)

    AdbKurucz is a class for Kurucz line list.

    Attributes:
        nurange: nu range [min,max] (cm-1)
        nu_lines (nd array):      line center (cm-1) (#NOT frequency in (s-1))
        dev_nu_lines (jnp array): line center (cm-1) in device
        Sij0 (nd array): line strength at T=Tref (cm)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient in (s-1)
        elower (jnp array): the lower state energy (cm-1)
        eupper (jnp array): the upper state energy (cm-1)
        gupper: (jnp array): upper statistical weight
        jlower (jnp array): lower J (rotational quantum number, total angular momentum)
        jupper (jnp array): upper J
        QTmask (jnp array): identifier of species for Q(T)
        ielem (jnp array):  atomic number (e.g., Fe=26)
        iion (jnp array):  ionized level (e.g., neutral=1, singly ionized=2, etc.)
        gamRad (jnp array): log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta (jnp array): log of gamma of Stark damping (s-1)
        vdWdamp (jnp array):  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
    """

    def __init__(
        self,
        path,
        nurange=[-np.inf, np.inf],
        margin=0.0,
        crit=0.0,
        Irwin=False,
        gpu_transfer=True,
        vmr_fraction=None,
    ):
        """Atomic database for Kurucz line list "gf????.all".

        Args:
            path: path for linelists (gf????.all) downloaded from the Kurucz web page
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
            crit: line strength lower limit for extraction
            Irwin: Use Irwin (1981) for Fe I; other species use Barklem & Collet (2016).
            gpu_transfer: If True, generate JAX line arrays at initialization.
            vmr_fraction: VMR fractions of H, He, and H2, in that order. Defaults to [0.0, 0.16, 0.84].

        Note:
            (written with reference to moldb.py, but without using feather format)
        """

        self.kurucz_file = pathlib.Path(path).expanduser()
        print("Reading Kurucz file")
        self._initialize(
            read_kurucz(self.kurucz_file), nurange, margin, crit, Irwin,
            gpu_transfer, vmr_fraction,
        )

    @classmethod
    def from_radis(
        cls, species, nurange, *, local_databases=None, cache=True,
        databank_name="ExoJAX-Kurucz-{molecule}", engine="pytables", margin=0.0,
        crit=0.0, Irwin=False, gpu_transfer=True, vmr_fraction=None,
    ):
        """Download or reuse one Kurucz species through RADIS.

        Args:
            species: Spectroscopic species, for example ``"Fe_I"`` or ``"Fe_II"``.
            nurange: Finite positive wavenumber interval or grid in cm-1.
            local_databases: RADIS cache directory, or its configured default.
            cache: RADIS cache policy; True reuses cached data.
            databank_name: RADIS registration name; ``{molecule}`` is replaced
                with the species. Use a distinct name for a separate cache.
            engine: RADIS cache engine, ``"pytables"`` or ``"vaex"``.
            margin: Extra wavenumber coverage in cm-1.
            crit: Nonnegative reference line-strength cutoff in cm.
            Irwin: Use Irwin (1981) for Fe I only.
            gpu_transfer: Generate JAX line arrays immediately if True.
            vmr_fraction: VMR fractions of H, He, and H2, in that order.

        Returns:
            An AdbKurucz with ExoJAX partition functions and broadening.

        Notes:
            RADIS controls downloads and cache registration. ``provenance``
            records the species, RADIS version, and returned cache paths.
            RADIS line positions and A coefficients are retained; its air/vacuum
            conversion can differ from the local-file constructor.
        """
        from exojax.database.kurucz._radis import (
            transitions_from_dataframe, validate_options, validate_species,
        )

        nurange = validate_options(nurange, margin, crit, vmr_fraction)
        ielem, iion = validate_species(species)
        if engine not in ("pytables", "vaex"):
            raise ValueError("engine must be 'pytables' or 'vaex'.")
        dataframe, local_paths = fetch_kurucz_dataframe(
            species,
            nurange=[max(0.0, nurange[0] - margin), nurange[1] + margin],
            local_databases=local_databases,
            databank_name=databank_name,
            engine=engine,
            cache=cache,
        )
        instance = cls.__new__(cls)
        instance.kurucz_file = None
        instance.provenance = {
            "backend": "radis",
            "radis_version": get_radis_version(),
            "species": species,
            "local_paths": [str(path) for path in local_paths],
        }
        instance._initialize(
            transitions_from_dataframe(dataframe, ielem, iion), nurange, margin,
            crit, Irwin, gpu_transfer, vmr_fraction,
        )
        return instance

    def _initialize(self, transitions, nurange, margin, crit, Irwin,
                    gpu_transfer, vmr_fraction):
        """Initialize either reader's normalized host transitions."""
        self.dbtype = "kurucz"
        self.Irwin = Irwin
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit
        if vmr_fraction is None:
            self.vmrH, self.vmrHe, self.vmrHH = [
                0.0,
                0.16,
                0.84,
            ]  # typical quasi-"solar-fraction"
        else:
            self.vmrH, self.vmrHe, self.vmrHH = vmr_fraction

        (
            self._A,
            self.nu_lines,
            self._elower,
            self._eupper,
            self._gupper,
            self._jlower,
            self._jupper,
            self._ielem,
            self._iion,
            self._gamRad,
            self._gamSta,
            self._vdWdamp,
        ) = transitions

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = load_pf_Barklem2016()  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(
            self.pfdat.iloc[:, 1:].to_numpy(dtype=float)
        )  # grid Q vs T vs Species
        self.Tref = Tref_original
        self.QTref_284 = np.array(
            interp_QT_284(Tref_original, self.T_gQT, self.gQT_284species, self.Irwin)
        )
        # identify index of QT grid (gQT) for each line
        self._QTmask = self.make_QTmask(self._ielem, self._iion)

        # Line strength: input shoud be ndarray not jnp array
        self.Sij0 = line_strength_atom(
            self._A,
            self._gupper,
            self.nu_lines,
            self._elower,
            self.QTref_284,
            self._QTmask,
        )  # 211013

        ### MASKING ###
        mask = (
            (self.nu_lines > self.nurange[0] - self.margin)
            * (self.nu_lines < self.nurange[1] + self.margin)
            * (self.Sij0 > self.crit)
        )

        self.masking(mask)
        _set_atomic_metadata(self)
        if gpu_transfer:
            self.generate_jnp_arrays()

    def masking(self, mask):
        """Select lines and metadata, refreshing existing JAX arrays.

        Args:
            mask: Boolean mask for the current lines.
        """
        _mask_atomic_lines(self, mask)

        if len(self.nu_lines) < 1:
            warn_msg = (
                "Warning: no lines are selected. Check the inputs to moldb.AdbKurucz."
            )
            warnings.warn(warn_msg, UserWarning)

    def generate_jnp_arrays(self):
        """Generate JAX line arrays and metadata for the current selection."""
        _generate_atomic_jnp_arrays(self)

    @property
    def line_masses(self):
        """Atomic masses for the current line selection, in amu."""
        return self.atomicmass

    def qr_interp_lines(self, T, Tref):
        """Return Q(T)/Q(Tref) for the current line selection."""
        return qr_interp_lines(
            T,
            Tref,
            self.T_gQT,
            self.gQT_284species,
            self._QTmask,
            getattr(self, "Irwin", False),
        )

    def Atomic_gQT(self, atomspecies):
        """Select grid of partition function especially for the species of
        interest.

        Args:
            atomspecies: species e.g., "Fe 1", "Sr 2", etc.

        Returns:
            gQT: grid Q(T) for the species
        """
        atomspecies_Roman = (
            atomspecies.split(" ")[0] + "_" + "I" * int(atomspecies.split(" ")[-1])
        )
        gQT = self.gQT_284species[np.where(self.pfdat["T[K]"] == atomspecies_Roman)][0]
        return gQT

    def QT_interp(self, atomspecies, T):
        """Interpolate the selected partition function for an atomic species.

        Args:
            atomspecies: species e.g., "Fe 1"
            T: temperature

        Returns:
            Q(T): interpolated in jnp.array for the Atomic Species
        """
        if getattr(self, "Irwin", False) and atomspecies == "Fe 1":
            return partfn_Fe(T)
        gQT = self.Atomic_gQT(atomspecies)
        QT = jnp.interp(T, self.T_gQT, gQT)
        return QT

    def QT_interp_Irwin_Fe(self, T, atomspecies="Fe 1"):
        """interpolated partition function This function is for the exceptional
        case where you want to adopt partition functions of Irwin (1981) for Fe
        I (Other species are not yet implemented).

        Args:
            atomspecies: species e.g., "Fe 1"
            T: temperature

        Returns:
            Q(T): interpolated in jnp.array for the Atomic Species
        """
        #gQT = self.Atomic_gQT(atomspecies)
        QT = partfn_Fe(T)
        return QT

    def qr_interp(self, atomspecies, T):
        """Return the selected partition-function ratio for an atomic species.

        Args:
            T: temperature
            atomspecies: species e.g., "Fe 1"

        Returns:
            qr(T)=Q(T)/Q(Tref): interpolated in jnp.array
        """
        return self.QT_interp(atomspecies, T) / self.QT_interp(
            atomspecies, Tref_original
        )

    def qr_interp_Irwin_Fe(self, T, atomspecies="Fe 1"):
        """interpolated partition function ratio This function is for the
        exceptional case where you want to adopt partition functions of Irwin
        (1981) for Fe I (Other species are not yet implemented).

        Args:
            T: temperature
            atomspecies: species e.g., "Fe 1"

        Returns:
            qr(T)=Q(T)/Q(Tref): interpolated in jnp.array
        """
        return self.QT_interp_Irwin_Fe(T, atomspecies) / self.QT_interp_Irwin_Fe(
            Tref_original, atomspecies
        )

    def QT_interp_284(self, T):
        """(DEPRECATED) interpolated partition function of all 284 species.

        Args:
            T: temperature

        Returns:
            Q(T)*284: interpolated in jnp.array for all 284 Atomic Species
        """
        warn_msg = "Deprecated Use `atomll.interp_QT_284` instead"
        warnings.warn(warn_msg, FutureWarning)
        return interp_QT_284(
            T, self.T_gQT, self.gQT_284species, getattr(self, "Irwin", False)
        )

    def make_QTmask(self, ielem, iion):
        """Convert the species identifier to the index for Q(Tref) grid (gQT)
        for each line.

        Args:
            ielem:  atomic number (e.g., Fe=26)
            iion:  ionized level (e.g., neutral=1, singly)

        Returns:
            QTmask_sp:  array of index of Q(Tref) grid (gQT) for each line
        """

        def species_to_QTmask(ielem, iion):
            sp_Roman = PeriodicTable[ielem] + "_" + "I" * iion
            QTmask = np.where(self.pfdat["T[K]"] == sp_Roman)[0][0]
            return QTmask

        QTmask_sp = np.array(list(map(species_to_QTmask, ielem, iion))).astype("int")
        return QTmask_sp
