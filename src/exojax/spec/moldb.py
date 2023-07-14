"""Molecular database (MDB) class.

"""
import numpy as np
import jax.numpy as jnp
import pathlib
import vaex
import warnings
from exojax.spec import atomllapi, atomll
from exojax.utils.constants import Tref_original
from exojax.spec import api 
__all__ = ['AdbVald', 'AdbSepVald', 'AdbKurucz']

explanation_states = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
explanation_trans = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
warning_old_exojax = 'It seems that the hdf5 file for the transition file was created using the old version of exojax<1.1. Try again after removing '


class AdbVald(object):
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

        Note:
           For the first time to read the VALD line list, it is converted to HDF/vaex. After the second-time, we use the HDF5 format with vaex instead.
    """

    def __init__(self, path, nurange=[-np.inf, np.inf], margin=0.0, crit=0., Irwin=False, gpu_transfer=True, vmr_fraction=None):
        """Atomic database for VALD3 "Long format".

        Args:
          path: path for linelists downloaded from VALD3 with a query of "Long format" in the format of "Extract All", "Extract Stellar", or "Extract Element"
          nurange: wavenumber range list (cm-1) or wavenumber array
          margin: margin for nurange (cm-1)
          crit: line strength lower limit for extraction
          Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016
          gpu_transfer: tranfer data to jnp.array? 
          vmr_fraction: list of the vmr fractions of hydrogen, H2 molecule, helium. if None, typical quasi-"solar-fraction" will be applied. 

        Note:
          (written with reference to moldb.py, but without using feather format)
        """

        self.dbtype = "vald"

        # load args
        self.vald3_file = pathlib.Path(path).expanduser()  # VALD3 output
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit
        if vmr_fraction is None:
            self.vmrH, self.vmrHe, self.vmrHH = [0.0, 0.16, 0.84] #typical quasi-"solar-fraction"
        else:
            self.vmrH, self.vmrHe, self.vmrHH = vmr_fraction

        # load vald file
        print('Reading VALD file')
        if self.vald3_file.with_suffix('.hdf5').exists():
            valdd = vaex.open(self.vald3_file.with_suffix('.hdf5'))
        else:
            print(
                "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format.")
            valdd = atomllapi.read_ExAll(self.vald3_file)  # vaex.DataFrame
        pvaldd = valdd.to_pandas_df()  # pandas.DataFrame

        # compute additional transition parameters
        self._A, self.nu_lines, self._elower, self._eupper, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._gamRad, self._gamSta, self._vdWdamp = atomllapi.pickup_param(
            pvaldd)

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = atomllapi.load_pf_Barklem2016()  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(
            dtype=float))  # grid Q vs T vs Species
        self.QTref_284 = np.array(self.QT_interp_284(Tref_original))
        # identify index of QT grid (gQT) for each line
        self._QTmask = self.make_QTmask(self._ielem, self._iion)

        # Line strength: input shoud be ndarray not jnp array
        self.Sij0 = atomll.Sij0(self._A, self._gupper, self.nu_lines,
                                self._elower, self.QTref_284, self._QTmask, Irwin)  # 211013

        ### MASKING ###
        mask = (self.nu_lines > self.nurange[0]-self.margin)\
            * (self.nu_lines < self.nurange[1]+self.margin)\
            * (self.Sij0 > self.crit)

        self.masking(mask)
        if gpu_transfer:
            self.generate_jnp_arrays()
        
        # Compile atomic-specific data for each absorption line of interest
        ipccd = atomllapi.load_atomicdata()
        self.solarA = jnp.array(
            list(map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 4], self.ielem)))
        self.atomicmass = jnp.array(
            list(map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 5], self.ielem)))
        df_ionE = atomllapi.load_ionization_energies()
        self.ionE = jnp.array(
            list(map(atomllapi.pick_ionE, self.ielem, self.iion, [df_ionE, ] * len(self.ielem))))

    def masking(self, mask):
        """applying mask.

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

        if(len(self.nu_lines) < 1):
            warn_msg = "Warning: no lines are selected. Check the inputs to moldb.AdbVald."
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
        atomspecies_Roman = atomspecies.split(
            ' ')[0] + '_' + 'I'*int(atomspecies.split(' ')[-1])
        gQT = self.gQT_284species[np.where(
            self.pfdat['T[K]'] == atomspecies_Roman)][0]
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

    def QT_interp_Irwin_Fe(self, T, atomspecies='Fe 1'):
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
        QT = atomllapi.partfn_Fe(T)
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
        return self.QT_interp(atomspecies, T)/self.QT_interp(atomspecies, Tref_original)

    def qr_interp_Irwin_Fe(self, T, atomspecies='Fe 1'):
        """interpolated partition function ratio This function is for the
        exceptional case where you want to adopt partition functions of Irwin
        (1981) for Fe I (Other species are not yet implemented).

        Args:
           T: temperature
           atomspecies: species e.g., "Fe 1"

        Returns:
           qr(T)=Q(T)/Q(Tref): interpolated in jnp.array
        """
        return self.QT_interp_Irwin_Fe(T, atomspecies)/self.QT_interp_Irwin_Fe(Tref_original, atomspecies)

    def QT_interp_284(self, T):
        """interpolated partition function of all 284 species.

        Args:
           T: temperature

        Returns:
           Q(T)*284: interpolated in jnp.array for all 284 Atomic Species
        """
        list_gQT_eachspecies = self.gQT_284species.tolist()
        listofDA_gQT_eachspecies = list(
            map(lambda x: jnp.array(x), list_gQT_eachspecies))
        listofQT = list(map(lambda x: jnp.interp(
            T, self.T_gQT, x), listofDA_gQT_eachspecies))
        QT_284 = jnp.array(listofQT)
        return QT_284

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
            sp_Roman = atomllapi.PeriodicTable[ielem] + '_' + 'I'*iion
            QTmask = np.where(self.pfdat['T[K]'] == sp_Roman)[0][0]
            return QTmask
        QTmask_sp = np.array(
            list(map(species_to_QTmask, ielem, iion))).astype('int')
        return QTmask_sp


class AdbSepVald(object):
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
    """

    def __init__(self, adb):
        """Species-separated atomic database for VALD3.

        Args:
            adb: adb instance made by the AdbVald class, which stores the lines of all species together

        """
        self.nu_lines = atomll.sep_arr_of_sp(
            adb.nu_lines, adb, trans_jnp=False)
        self.QTmask = atomll.sep_arr_of_sp(adb.QTmask, adb, inttype=True).T[0]

        self.ielem = atomll.sep_arr_of_sp(adb.ielem, adb, inttype=True).T[0]
        self.iion = atomll.sep_arr_of_sp(adb.iion, adb, inttype=True).T[0]
        self.atomicmass = atomll.sep_arr_of_sp(adb.atomicmass, adb).T[0]
        self.ionE = atomll.sep_arr_of_sp(adb.ionE, adb).T[0]

        self.logsij0 = atomll.sep_arr_of_sp(adb.logsij0, adb)
        self.dev_nu_lines = atomll.sep_arr_of_sp(adb.dev_nu_lines, adb)
        self.elower = atomll.sep_arr_of_sp(adb.elower, adb)
        self.eupper = atomll.sep_arr_of_sp(adb.eupper, adb)
        self.gamRad = atomll.sep_arr_of_sp(adb.gamRad, adb)
        self.gamSta = atomll.sep_arr_of_sp(adb.gamSta, adb)
        self.vdWdamp = atomll.sep_arr_of_sp(adb.vdWdamp, adb)

        self.uspecies = atomll.get_unique_species(adb)
        self.N_usp = len(self.uspecies)
        self.L_max = self.nu_lines.shape[1]

        self.gQT_284species = adb.gQT_284species
        self.T_gQT = adb.T_gQT


class AdbKurucz(object):
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

    def __init__(self, path, nurange=[-np.inf, np.inf], margin=0.0, crit=0., Irwin=False, gpu_transfer=True, vmr_fraction=None):
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
            self.vmrH, self.vmrHe, self.vmrHH = [0.0, 0.16, 0.84] #typical quasi-"solar-fraction"
        else:
            self.vmrH, self.vmrHe, self.vmrHH = vmr_fraction

        # load kurucz file
        print('Reading Kurucz file')
        self._A, self.nu_lines, self._elower, self._eupper, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._gamRad, self._gamSta, self._vdWdamp = atomllapi.read_kurucz(
            self.kurucz_file)

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = atomllapi.load_pf_Barklem2016()  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(
            dtype=float))  # grid Q vs T vs Species
        self.QTref_284 = np.array(self.QT_interp_284(Tref_original))
        # identify index of QT grid (gQT) for each line
        self._QTmask = self.make_QTmask(self._ielem, self._iion)

        # Line strength: input shoud be ndarray not jnp array
        self.Sij0 = atomll.Sij0(self._A, self._gupper, self.nu_lines,
                                self._elower, self.QTref_284, self._QTmask, Irwin)  # 211013

        ### MASKING ###
        mask = (self.nu_lines > self.nurange[0]-self.margin)\
            * (self.nu_lines < self.nurange[1]+self.margin)\
            * (self.Sij0 > self.crit)

        self.masking(mask)
        if gpu_transfer:
            self.generate_jnp_arrays()
        
        # Compile atomic-specific data for each absorption line of interest
        ipccd = atomllapi.load_atomicdata()
        self.solarA = jnp.array(
            list(map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 4], self.ielem)))
        self.atomicmass = jnp.array(
            list(map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 5], self.ielem)))
        df_ionE = atomllapi.load_ionization_energies()
        self.ionE = jnp.array(
            list(map(atomllapi.pick_ionE, self.ielem, self.iion, [df_ionE, ] * len(self.ielem))))

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

        if(len(self.nu_lines) < 1):
            warn_msg = "Warning: no lines are selected. Check the inputs to moldb.AdbKurucz."
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
        atomspecies_Roman = atomspecies.split(
            ' ')[0] + '_' + 'I'*int(atomspecies.split(' ')[-1])
        gQT = self.gQT_284species[np.where(
            self.pfdat['T[K]'] == atomspecies_Roman)][0]
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

    def QT_interp_Irwin_Fe(self, T, atomspecies='Fe 1'):
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
        QT = atomllapi.partfn_Fe(T)
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
        return self.QT_interp(atomspecies, T)/self.QT_interp(atomspecies, Tref_original)

    def qr_interp_Irwin_Fe(self, T, atomspecies='Fe 1'):
        """interpolated partition function ratio This function is for the
        exceptional case where you want to adopt partition functions of Irwin
        (1981) for Fe I (Other species are not yet implemented).

        Args:
           T: temperature
           atomspecies: species e.g., "Fe 1"

        Returns:
           qr(T)=Q(T)/Q(Tref): interpolated in jnp.array
        """
        return self.QT_interp_Irwin_Fe(T, atomspecies)/self.QT_interp_Irwin_Fe(Tref_original, atomspecies)

    def QT_interp_284(self, T):
        """interpolated partition function of all 284 species.

        Args:
           T: temperature

        Returns:
           Q(T)*284: interpolated in jnp.array for all 284 Atomic Species
        """
        list_gQT_eachspecies = self.gQT_284species.tolist()
        listofDA_gQT_eachspecies = list(
            map(lambda x: jnp.array(x), list_gQT_eachspecies))
        listofQT = list(map(lambda x: jnp.interp(
            T, self.T_gQT, x), listofDA_gQT_eachspecies))
        QT_284 = jnp.array(listofQT)
        return QT_284

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
            sp_Roman = atomllapi.PeriodicTable[ielem] + '_' + 'I'*iion
            QTmask = np.where(self.pfdat['T[K]'] == sp_Roman)[0][0]
            return QTmask
        QTmask_sp = np.array(
            list(map(species_to_QTmask, ielem, iion))).astype('int')
        return QTmask_sp
