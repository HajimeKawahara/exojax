"""Atomic database (MDB) class."""

import pathlib
import warnings

import jax.numpy as jnp
import numpy as np

from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.pf import interp_QT_284
from exojax.database.core_atom.pf import partfn_Fe

from exojax.database.core_atom.io import read_kurucz
from exojax.database.core_atom.io import load_pf_Barklem2016
from exojax.database.core_atom.io import load_atomicdata
from exojax.database.core_atom.io import load_ionization_energies
from exojax.database.core_atom.io import pick_ionE
from exojax.database.core_atom.io import PeriodicTable
# from exojax.database.core_atom.io import load_ionization_energies  # Duplicate import removed
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
            Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016
            gpu_transfer: tranfer data to jnp.array?
            vmr_fraction: list of the vmr fractions of hydrogen, H2 molecule, helium. if None, typical quasi-"solar-fraction" will be applied.

        Note:
            (written with reference to moldb.py, but without using feather format)
        """

        self.dbtype = "kurucz"

        # load args
        self.kurucz_file = pathlib.Path(path).expanduser()
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

        # load kurucz file
        print("Reading Kurucz file")
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
        ) = read_kurucz(self.kurucz_file)

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = load_pf_Barklem2016()  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(
            self.pfdat.iloc[:, 1:].to_numpy(dtype=float)
        )  # grid Q vs T vs Species
        self.Tref = Tref_original
        self.QTref_284 = np.array(
            interp_QT_284(Tref_original, self.T_gQT, self.gQT_284species)
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
            Irwin,
        )  # 211013

        ### MASKING ###
        mask = (
            (self.nu_lines > self.nurange[0] - self.margin)
            * (self.nu_lines < self.nurange[1] + self.margin)
            * (self.Sij0 > self.crit)
        )

        self.masking(mask)
        if gpu_transfer:
            self.generate_jnp_arrays()

        # Compile atomic-specific data for each absorption line of interest
        ipccd = load_atomicdata()
        self.solarA = jnp.array(
            list(map(lambda x: ipccd[ipccd["ielem"] == x].iat[0, 4], self.ielem))
        )
        self.atomicmass = jnp.array(
            list(map(lambda x: ipccd[ipccd["ielem"] == x].iat[0, 5], self.ielem))
        )
        df_ionE = load_ionization_energies()
        self.ionE = jnp.array(
            list(
                map(
                    pick_ionE,
                    self.ielem,
                    self.iion,
                    [
                        df_ionE,
                    ]
                    * len(self.ielem),
                )
            )
        )

    def masking(self, mask):
        """applying mask

        Args:
            mask: mask to be applied. self.mask is updated.

        """
        # numpy float 64 Do not convert them jnp array
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
            warn_msg = (
                "Warning: no lines are selected. Check the inputs to moldb.AdbKurucz."
            )
            warnings.warn(warn_msg, UserWarning)

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        # jnp arrays
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
        """interpolated partition function The partition functions of Barklem &
        Collet (2016) are adopted.

        Args:
            atomspecies: species e.g., "Fe 1"
            T: temperature

        Returns:
            Q(T): interpolated in jnp.array for the Atomic Species
        """
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
        """interpolated partition function ratio The partition functions of
        Barklem & Collet (2016) are adopted.

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
        return interp_QT_284(T, self.T_gQT, self.gQT_284species)

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
