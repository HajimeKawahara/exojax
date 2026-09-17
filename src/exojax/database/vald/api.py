"""Atomic database (ADB) class."""

import pathlib
import warnings
import jax.numpy as jnp
import numpy as np
from exojax.database.core_atom._arrays import (
    _generate_atomic_jnp_arrays,
    _mask_atomic_lines,
    _set_atomic_metadata,
)
from exojax.database.core_atom.io import load_pf_Barklem2016
from exojax.database.core_atom.io import PeriodicTable
from exojax.database.core_atom.io import _normalize_vald_engine
from exojax.database.core_atom.io import _vald_cache_path
from exojax.database.core_atom.io import read_ExAll
from exojax.database.core_atom.io import pickup_param

from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.pf import interp_QT_284
from exojax.database.core_atom.pf import partfn_Fe
from exojax.database.core_atom.pf import qr_interp_lines
from exojax.database.core_atom.misc import get_unique_species
from exojax.database.core_atom.misc import sep_arr_of_sp

from exojax.utils.constants import Tref_original

__all__ = ["AdbVald", "AdbSepVald"]

explanation_states = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
explanation_trans = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
warning_old_exojax = "It seems that the hdf5 file for the transition file was created using the old version of exojax<1.1. Try again after removing "


def _load_vald_dataframe(vald3_file, engine):
    """Load a VALD line list from its cache, creating the cache if needed."""
    engine = _normalize_vald_engine(engine)
    cache_path = _vald_cache_path(vald3_file, engine)

    if cache_path.exists():
        if engine == "vaex":
            import vaex

            return vaex.open(cache_path).to_pandas_df()

        import pandas as pd

        return pd.read_hdf(cache_path, key="dat")

    print(
        f"Note: Couldn't find the {engine} cache. "
        f"Converting the VALD line list to {cache_path.name}."
    )
    dataframe = read_ExAll(vald3_file, engine=engine)
    if engine == "vaex":
        return dataframe.to_pandas_df()
    return dataframe


class AdbVald:
    """atomic database from VALD3 (http://vald.astro.uu.se/)

    AdbVald is a class for VALD3.

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
        solarA (jnp array): solar abundance (log10 of number density in the Sun)
        atomicmass (jnp array): atomic mass (amu)
        ionE (jnp array): ionization potential (eV)
        gamRad (jnp array): log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta (jnp array): log of gamma of Stark damping (s-1)
        vdWdamp (jnp array):  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
        gQT_284species (jnp array): partition function grid of 284 species
        T_gQT (jnp array): temperatures in the partition function grid
        QTref_284 (jnp array): partition function at the reference temperature Q(Tref), for 284 species

        Note:
            On first use, the VALD line list is converted to a backend-specific
            HDF5 cache. PyTables uses ``.h5`` and vaex uses ``.hdf5``. Later
            reads reuse that cache.
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
        engine="pytables",
    ):
        """Atomic database for VALD3 "Long format".

        Args:
            path: path for linelists downloaded from VALD3 with a query of "Long format" in the format of "Extract All", "Extract Stellar", or "Extract Element"
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
            crit: line strength lower limit for extraction
            Irwin: Use Irwin (1981) for Fe I; other species use Barklem & Collet (2016).
            gpu_transfer: If True, generate JAX line arrays at initialization.
            vmr_fraction: VMR fractions of H, He, and H2, in that order. Defaults to [0.0, 0.16, 0.84].
            engine: ``"pytables"`` (default), its legacy alias ``"pandas"``,
                or the optional ``"vaex"`` backend.

        Note:
            (written with reference to moldb.py, but without using feather format)
        """

        self.dbtype = "vald"
        self.Irwin = Irwin

        # load args
        self.vald3_file = pathlib.Path(path).expanduser()  # VALD3 output
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit
        self.engine = _normalize_vald_engine(engine)
        if vmr_fraction is None:
            self.vmrH, self.vmrHe, self.vmrHH = [
                0.0,
                0.16,
                0.84,
            ]  # typical quasi-"solar-fraction"
        else:
            self.vmrH, self.vmrHe, self.vmrHH = vmr_fraction

        # load vald file
        print("Reading VALD file")
        pvaldd = _load_vald_dataframe(self.vald3_file, self.engine)

        # compute additional transition parameters
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
        ) = pickup_param(pvaldd)

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
                "Warning: no lines are selected. Check the inputs to moldb.AdbVald."
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
        gQT = self.Atomic_gQT(atomspecies)
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
            iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)

        Returns:
            QTmask_sp:  array of index of Q(Tref) grid (gQT) for each line
        """

        def species_to_QTmask(ielem, iion):
            sp_Roman = PeriodicTable[ielem] + "_" + "I" * iion
            QTmask = np.where(self.pfdat["T[K]"] == sp_Roman)[0][0]
            return QTmask

        QTmask_sp = np.array(list(map(species_to_QTmask, ielem, iion))).astype("int")
        return QTmask_sp


class AdbSepVald:
    """atomic database from VALD3 with an additional axis for separating each
    species (atom or ion)

    AdbSepVald is a class for VALD3.

    Attributes:
        nu_lines (nd array):      line center (cm-1) (#NOT frequency in (s-1))
        dev_nu_lines (jnp array): line center (cm-1) in device
        logsij0 (jnp array): log line strength at T=Tref
        elower (jnp array): the lower state energy (cm-1)
        eupper (jnp array): the upper state energy (cm-1)
        QTmask (jnp array): identifier of species for Q(T)
        ielem (jnp array):  atomic number (e.g., Fe=26)
        iion (jnp array):  ionized level (e.g., neutral=1, singly ionized=2, etc.)
        atomicmass (jnp array): atomic mass (amu)
        ionE (jnp array): ionization potential (eV)
        gamRad (jnp array): log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta (jnp array): log of gamma of Stark damping (s-1)
        vdWdamp (jnp array):  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
        uspecies (jnp array): unique combinations of ielem and iion [N_species x 2(ielem and iion)]
        N_usp (int): number of species (atoms and ions)
        L_max (int): maximum number of spectral lines for a single species
        gQT_284species (jnp array): partition function grid of 284 species
        T_gQT (jnp array): temperatures in the partition function grid
        QTref_284 (jnp array): partition function at the reference temperature Q(Tref), for 284 species
        Tref (float): reference temperature
    """

    def __init__(self, adb):
        """Species-separated atomic database for VALD3.

        Args:
            adb: adb instance made by the AdbVald class, which stores the lines of all species together

        """
        self.nu_lines = sep_arr_of_sp(adb.nu_lines, adb, trans_jnp=False)
        self.QTmask = sep_arr_of_sp(adb.QTmask, adb, inttype=True).T[0]

        self.ielem = sep_arr_of_sp(adb.ielem, adb, inttype=True).T[0]
        self.iion = sep_arr_of_sp(adb.iion, adb, inttype=True).T[0]
        self.atomicmass = sep_arr_of_sp(adb.atomicmass, adb).T[0]
        self.ionE = sep_arr_of_sp(adb.ionE, adb).T[0]

        self.logsij0 = sep_arr_of_sp(adb.logsij0, adb)
        self.dev_nu_lines = sep_arr_of_sp(adb.dev_nu_lines, adb)
        self.elower = sep_arr_of_sp(adb.elower, adb)
        self.eupper = sep_arr_of_sp(adb.eupper, adb)
        self.gamRad = sep_arr_of_sp(adb.gamRad, adb)
        self.gamSta = sep_arr_of_sp(adb.gamSta, adb)
        self.vdWdamp = sep_arr_of_sp(adb.vdWdamp, adb)

        self.uspecies = get_unique_species(adb)
        self.N_usp = len(self.uspecies)
        self.L_max = self.nu_lines.shape[1]

        self.gQT_284species = adb.gQT_284species
        self.T_gQT = adb.T_gQT
        self.QTref_284 = adb.QTref_284
        self.Tref = adb.Tref
        self.Irwin = getattr(adb, "Irwin", False)
