"""Molecular database (MDB) class using a common API w/ RADIS = (CAPI), will be renamed.

* MdbExomol is the MDB for ExoMol
* MdbHit is the MDB for HITRAN or HITEMP
"""
import os
import warnings
from os.path import abspath, exists
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import pathlib
import vaex
from exojax.spec import hapi, exomolapi, exomol, atomllapi, atomll, hitranapi
from exojax.spec.hitran import gamma_natural as gn
from exojax.utils.constants import hcperk, Tref
from exojax.utils.molname import e2s

# currently use radis add/common-api branch
from radis.api.exomolapi import MdbExomol as CapiMdbExomol  #MdbExomol in the common API
from radis.api.hitempapi import HITEMPDatabaseManager
from radis.api.hdf5 import update_pytables_to_vaex
from radis.db.classes import get_molecule
from radis.levels.partfunc import PartFuncHAPI

__all__ = ['MdbExomol', 'MdbHit', 'AdbVald', 'AdbKurucz']

explanation_states = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
explanation_trans = "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster."
warning_old_exojax = 'It seems that the hdf5 file for the transition file was created using the old version of exojax<1.1. Try again after removing '


class MdbExomol(CapiMdbExomol):
    """molecular database of ExoMol.
    
    MdbExomol is a class for ExoMol.

    Attributes:
        simple_molecule_name: simple molecule name
        nurange: nu range [min,max] (cm-1)
        nu_lines (nd array): line center (cm-1)
        Sij0 (nd array): line strength at T=Tref (cm)
        
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient
        gamma_natural (DataFrame or jnp array): gamma factor of the natural broadening
        elower (DataFrame or jnp array): the lower state energy (cm-1)
        gpp (DataFrame or jnp array): statistical weight
        jlower (DataFrame or jnp array): J_lower
        jupper (DataFrame or jnp array): J_upper
        n_Texp (DataFrame or jnp array): temperature exponent
        dev_nu_lines (jnp array): line center in device (cm-1)
        alpha_ref (jnp array): alpha_ref (gamma0), Lorentzian half-width at reference temperature and pressure in cm-1/bar
        n_Texp_def: default temperature exponent in .def file, used for jlower not given in .broad
        alpha_ref_def: default alpha_ref (gamma0) in .def file, used for jlower not given in .broad
    """
    __slots__ = [
        "Sij0",
        "logsij0",
        "nu_lines",
        "A",
        "elower",
        "eupper",
        "gupper",
        "jlower",
        "jupper",
    ]

    def __init__(self,
                 path,
                 nurange=[-np.inf, np.inf],
                 margin=0.0,
                 crit=0.,
                 Ttyp=1000.,
                 bkgdatm='H2',
                 broadf=True,
                 gpu_transfer=True,
                 local_databases="./"):
        """Molecular database for Exomol form.

        Args:
           path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
           nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           Ttyp: typical temperature to calculate Sij(T) used in crit
           bkgdatm: background atmosphere for broadening. e.g. H2, He,
           broadf: if False, the default broadening parameters in .def file is used
           
        Note:
           The trans/states files can be very large. For the first time to read it, we convert it to HDF/vaex. After the second-time, we use the HDF5 format with vaex instead.
        """
        self.path = pathlib.Path(path).expanduser()
        self.exact_molecule_name = self.path.parents[0].stem
        self.database = str(self.path.stem)
        self.bkgdatm = bkgdatm
        #molecbroad = self.exact_molecule_name + '__' + self.bkgdatm

        self.Ttyp = Ttyp
        self.broadf = broadf
        self.simple_molecule_name = e2s(self.exact_molecule_name)

        wavenum_min, wavenum_max = np.min(nurange), np.max(nurange)
        if wavenum_min == -np.inf:
            wavenum_min = None
        if wavenum_max == np.inf:
            wavenum_max = None

        super().__init__(str(self.path),
                         local_databases=local_databases,
                         molecule=self.simple_molecule_name,
                         name="EXOMOL-{molecule}",
                         nurange=[wavenum_min, wavenum_max],
                         engine="vaex",
                         margin=margin,
                         crit=crit,
                         bkgdatm=self.bkgdatm,
                         cache=True,
                         skip_optional_data=True)

        self.crit = crit
        self.QTtyp = np.array(self.QT_interp(self.Ttyp))

        # Get cache files to load :
        mgr = self.get_datafile_manager()
        local_files = [mgr.cache_file(f) for f in self.trans_file]

        # Load them:
        df = self.load(
            local_files,
            columns=[k for k in self.__slots__ if k not in ["logsij0"]],
            #lower_bound=([("nu_lines", wavenum_min)] if wavenum_min else []) +
            lower_bound=([("Sij0", 0.0)]),
            #upper_bound=([("nu_lines", wavenum_max)] if wavenum_max else []),
            output="vaex")

        load_mask = self.compute_load_mask(df)
        self.get_values_from_dataframes(df[load_mask])
        self.compute_broadening(self.jlower, self.jupper)

        self.gamma_natural = gn(self.A)

        if gpu_transfer:
            self.generate_jnp_arrays()

    def compute_load_mask(self, df):
        load_mask = (df.nu_lines > self.nurange[0]-self.margin) \
                    * (df.nu_lines < self.nurange[1]+self.margin)
        load_mask = load_mask * (self.get_Sij_typ(df.Sij0, df.elower,
                                                  df.nu_lines) > self.crit)
        return load_mask

    def get_values_from_dataframes(self, df):

        if isinstance(df, vaex.dataframe.DataFrameLocal):
            self.A = df.A.values
            self.nu_lines = df.nu_lines.values
            self.elower = df.elower.values
            self.jlower = df.jlower.values
            self.jupper = df.jupper.values
            self.Sij0 = df.Sij0.values
            self.gpp = df.gup.values
        else:
            raise ValueError("Use vaex dataframe as input.")

    def get_Sij_typ(self, Sij0_in, elower_in, nu_in):
        """compute Sij at typical temperature self.Ttyp.

        Args:
           Sij0_in : line strength at Tref
           elower_in: elower
           nu_in: wavenumber bin

        Returns:
           Sij at Ttyp
        """
        return Sij0_in * self.QTref / self.QTtyp \
            * np.exp(-hcperk*elower_in * (1./self.Ttyp - 1./Tref)) \
            * np.expm1(-hcperk*nu_in/self.Ttyp) / np.expm1(-hcperk*nu_in/Tref)

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        # jnp arrays
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.Sij0))
        self.gamma_natural = jnp.array(self.gamma_natural)
        self.A = jnp.array(self.A)
        self.elower = jnp.array(self.elower)
        self.gpp = jnp.array(self.gpp)
        self.jlower = jnp.array(self.jlower, dtype=int)
        self.jupper = jnp.array(self.jupper, dtype=int)
        self.alpha_ref = jnp.array(self.alpha_ref)
        self.n_Texp = jnp.array(self.n_Texp)

    def QT_interp(self, T):
        """interpolated partition function.

        Args:
           T: temperature

        Returns:
           Q(T) interpolated in jnp.array
        """
        return jnp.interp(T, self.T_gQT, self.gQT)

    def qr_interp(self, T):
        """interpolated partition function ratio.

        Args:
           T: temperature

        Returns:
           qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        return self.QT_interp(T) / self.QT_interp(Tref)


#copied from moldb, should move it.
def search_molecid(molec):
    """molec id from molec (source table name) of HITRAN/HITEMP.

    Args:
       molec: source table name

    Return:
       int: molecid (HITRAN molecular id)
    """
    try:
        hitf = molec.split('_')
        molecid = int(hitf[0])
        return molecid
    except:
        raise ValueError('Define molecid by yourself.')


class MdbHit(HITEMPDatabaseManager):
    """molecular database of HITEMP.

    Attributes:
        simple_molecule_name: simple molecule name
        nurange: nu range [min,max] (cm-1)
        nu_lines (nd array): line center (cm-1)
        Sij0 (nd array): line strength at T=Tref (cm)
        dev_nu_lines (jnp array): line center in device (cm-1)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient
        gamma_natural (jnp array): gamma factor of the natural broadening
        gamma_air (jnp array): gamma factor of air pressure broadening
        gamma_self (jnp array): gamma factor of self pressure broadening
        elower (jnp array): the lower state energy (cm-1)
        gpp (jnp array): statistical weight
        n_air (jnp array): air temperature exponent
    """
    def __init__(self,
                 path,
                 nurange=[-np.inf, np.inf],
                 margin=0.0,
                 crit=0.,
                 Ttyp=1000.,
                 isotope=None,
                 gpu_transfer=True):
        """Molecular database for HITRAN/HITEMP form.

        Args:
           path: path for HITRAN/HITEMP par file
           nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           Ttyp: typical temperature to calculate Sij(T) used in crit
           isotope: None= use all isotopes. 
           gpu_transfer: tranfer data to jnp.array?
        """

        if ("hit" in path and path[-4:] == ".bz2"):
            path = path[:-4]
            print('Warning: path changed (.bz2 removed):', path)
        if ("HITEMP" in path and path[-4:] == ".par"):
            path = path + '.bz2'
            print('Warning: path changed (.bz2 added):', path)

        self.path = pathlib.Path(path).expanduser()
        self.molecid = search_molecid(str(self.path.stem))
        self.simple_molecule_name = get_molecule(self.molecid)

        #numinf, numtag = hitranapi.read_path(self.path)
        self.crit = crit
        self.Ttyp = Ttyp
        self.margin = margin
        self.nurange = [np.min(nurange), np.max(nurange)]
        load_wavenum_min = self.nurange[0] - self.margin
        load_wavenum_max = self.nurange[1] + self.margin

        super().__init__(
            molecule="CO",
            name="HITEMP-{molecule}",
            local_databases="",
            engine="default",
            verbose=True,
            chunksize=100000,
            parallel=True,
        )

        # Get list of all expected local files for this database:
        local_files, urlnames = self.get_filenames()

        # Get missing files
        download_files = self.get_missing_files(local_files)
        download_files = self.keep_only_relevant(download_files,
                                                 load_wavenum_min,
                                                 load_wavenum_max)
        # do not re-download files if they exist in another format :

        converted = []
        for f in download_files:
            if exists(f.replace(".hdf5", ".h5")):
                update_pytables_to_vaex(f.replace(".hdf5", ".h5"))
                converted.append(f)
            download_files = [f for f in download_files if f not in converted]
        # do not re-download remaining files that exist. Let user decide what to do.
        # (download & re-parsing is a long solution!)
        download_files = [
            f for f in download_files if not exists(f.replace(".hdf5", ".h5"))
        ]

        # Download files
        if len(download_files) > 0:
            if urlnames is None:
                urlnames = self.fetch_urlnames()
            filesmap = dict(zip(local_files, urlnames))
            download_urls = [filesmap[k] for k in download_files]
            self.download_and_parse(download_urls, download_files)

        # Register
        if not self.is_registered():
            self.register()

        clean_cache_files = True
        if len(download_files) > 0 and clean_cache_files:
            self.clean_download_files()

        # Load and return
        files_loaded = self.keep_only_relevant(local_files, load_wavenum_min,
                                               load_wavenum_max)
        isotope = None
        columns = None,
        output = "vaex"

        if isotope and type(isotope) == int:
            isotope = str(isotope)

        df = self.load(
            files_loaded,  # filter other files,
            columns=columns,
            within=[("iso", isotope)] if isotope is not None else [],
            # for relevant files, get only the right range :
            lower_bound=[("wav", load_wavenum_min)]
            if self.nurange[0] is not None else [],
            upper_bound=[("wav", load_wavenum_max)]
            if self.nurange[1] is not None else [],
            output=output,
        )

        #M = get_molecule_identifier(molec)
        self.isoid = df.iso
        self.uniqiso = np.unique(df.iso.values)

        load_mask = None
        for iso in self.uniqiso:
            Q = PartFuncHAPI(self.molecid, iso)
            QTref = Q.at(T=Tref)
            QTtyp = Q.at(T=Ttyp)
            load_mask = self.compute_load_mask(df, QTref, QTtyp, load_mask)
        self.get_values_from_dataframes(df[load_mask])

    def get_Sij_typ(self, Sij0_in, elower_in, nu_in, QTref, QTtyp):
        """compute Sij at typical temperature self.Ttyp.

        Args:
           Sij0_in : line strength at Tref
           elower_in: elower
           nu_in: wavenumber bin

        Returns:
           Sij at Ttyp
        """
        return Sij0_in * QTref / QTtyp \
            * np.exp(-hcperk*elower_in * (1./self.Ttyp - 1./Tref)) \
            * np.expm1(-hcperk*nu_in/self.Ttyp) / np.expm1(-hcperk*nu_in/Tref)

    def compute_load_mask(self, df, QTref, QTtyp, load_mask):
        wav_mask = (df.wav > self.nurange[0]-self.margin) \
                    * (df.wav < self.nurange[1]+self.margin)
        intensity_mask = (self.get_Sij_typ(df.int, df.El, df.wav, QTref, QTtyp)
                          > self.crit)
        if load_mask is None:
            return wav_mask * intensity_mask
        else:
            return load_mask * wav_mask * intensity_mask

    def get_values_from_dataframes(self, df):
        if isinstance(df, vaex.dataframe.DataFrameLocal):
            self.nu_lines = df.wav.values
            self.Sij0 = df.int.values
            self.delta_air = df.Pshft.values
            self.isoid = df.iso.values
            self.uniqiso = np.unique(self.isoid)
            self.A = df.A.values
            self.n_air = df.Tdpair.values
            self.gamma_air = df.airbrd.values
            self.gamma_self = df.selbrd.values
            self.elower = df.El.values
            self.gpp = df.gp.values
        else:
            raise ValueError("Use vaex dataframe as input.")

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        self.nu_lines = jnp.array(self.nu_lines)
        self.Sij0 = jnp.array(self.Sij0)
        self.delta_air = jnp.array(self.delta_air)
        self.isoid = jnp.array(self.isoid)
        self.uniqiso = jnp.array(self.uniqiso)
        self.A = jnp.array(self.A)
        self.n_air = jnp.array(self.n_air)
        self.gamma_air = jnp.array(self.gamma_air)
        self.gamma_self = jnp.array(self.gamma_self)
        self.elower = jnp.array(self.elower)
        self.gpp = jnp.array(self.gpp)

    def QT_iso_interp(self, idx, T):
        """interpolated partition function.

        Args:
           idx: index for HITRAN isotopologue number
           T: temperature

        Returns:
           Q(idx, T) interpolated in jnp.array
        """
        return jnp.interp(T, self.T_gQT[idx], self.gQT[idx])

    def qr_iso_interp(self, idx, T):
        """interpolated partition function ratio.

        Args:
           idx: index for HITRAN isotopologue number
           T: temperature

        Returns:
           qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        return self.QT_iso_interp(idx, T) / self.QT_iso_interp(idx, self.Tref)


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
    def __init__(self,
                 path,
                 nurange=[-np.inf, np.inf],
                 margin=0.0,
                 crit=0.,
                 Irwin=False,
                 gpu_transfer=True):
        """Atomic database for VALD3 "Long format".

        Args:
          path: path for linelists downloaded from VALD3 with a query of "Long format" in the format of "Extract All", "Extract Stellar", or "Extract Element"
          nurange: wavenumber range list (cm-1) or wavenumber array
          margin: margin for nurange (cm-1)
          crit: line strength lower limit for extraction
          Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016
          gpu_transfer: tranfer data to jnp.array? 

        Note:
          (written with reference to moldb.py, but without using feather format)
        """

        # load args
        self.vald3_file = pathlib.Path(path).expanduser()  # VALD3 output
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit

        # load vald file
        print('Reading VALD file')
        if self.vald3_file.with_suffix('.hdf5').exists():
            valdd = vaex.open(self.vald3_file.with_suffix('.hdf5'))
        else:
            print(
                "Note: Couldn't find the hdf5 format. We convert data to the hdf5 format."
            )
            valdd = atomllapi.read_ExAll(self.vald3_file)  # vaex.DataFrame
        pvaldd = valdd.to_pandas_df()  # pandas.DataFrame

        # compute additional transition parameters
        self._A, self.nu_lines, self._elower, self._eupper, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._gamRad, self._gamSta, self._vdWdamp = atomllapi.pickup_param(
            pvaldd)

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = atomllapi.load_pf_Barklem2016(
        )  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(
            dtype=float))  # grid Q vs T vs Species
        self.QTref_284 = np.array(self.QT_interp_284(Tref))
        # identify index of QT grid (gQT) for each line
        self._QTmask = self.make_QTmask(self._ielem, self._iion)

        # Line strength: input shoud be ndarray not jnp array
        self.Sij0 = atomll.Sij0(self._A, self._gupper, self.nu_lines,
                                self._elower, self.QTref_284, self._QTmask,
                                Irwin)  # 211013

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
            list(
                map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 4],
                    self.ielem)))
        self.atomicmass = jnp.array(
            list(
                map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 5],
                    self.ielem)))
        df_ionE = atomllapi.load_ionization_energies()
        self.ionE = jnp.array(
            list(
                map(atomllapi.pick_ionE, self.ielem, self.iion, [
                    df_ionE,
                ] * len(self.ielem))))

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
        atomspecies_Roman = atomspecies.split(' ')[0] + '_' + 'I' * int(
            atomspecies.split(' ')[-1])
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
        return self.QT_interp(atomspecies, T) / self.QT_interp(
            atomspecies, Tref)

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
        return self.QT_interp_Irwin_Fe(
            T, atomspecies) / self.QT_interp_Irwin_Fe(Tref, atomspecies)

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
        listofQT = list(
            map(lambda x: jnp.interp(T, self.T_gQT, x),
                listofDA_gQT_eachspecies))
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
            sp_Roman = atomllapi.PeriodicTable[ielem] + '_' + 'I' * iion
            QTmask = np.where(self.pfdat['T[K]'] == sp_Roman)[0][0]
            return QTmask

        QTmask_sp = np.array(list(map(species_to_QTmask, ielem,
                                      iion))).astype('int')
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
        self.nu_lines = atomll.sep_arr_of_sp(adb.nu_lines,
                                             adb,
                                             trans_jnp=False)
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
    def __init__(self,
                 path,
                 nurange=[-np.inf, np.inf],
                 margin=0.0,
                 crit=0.,
                 Irwin=False,
                 gpu_transfer=True):
        """Atomic database for Kurucz line list "gf????.all".

        Args:
          path: path for linelists (gf????.all) downloaded from the Kurucz web page
          nurange: wavenumber range list (cm-1) or wavenumber array
          margin: margin for nurange (cm-1)
          crit: line strength lower limit for extraction
          Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016
          gpu_transfer: tranfer data to jnp.array? 

        Note:
          (written with reference to moldb.py, but without using feather format)
        """

        # load args
        self.kurucz_file = pathlib.Path(path).expanduser()
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.crit = crit

        # load kurucz file
        print('Reading Kurucz file')
        self._A, self.nu_lines, self._elower, self._eupper, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._gamRad, self._gamSta, self._vdWdamp = atomllapi.read_kurucz(
            self.kurucz_file)

        # load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = atomllapi.load_pf_Barklem2016(
        )  # Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(
            dtype=float))  # grid Q vs T vs Species
        self.QTref_284 = np.array(self.QT_interp_284(Tref))
        # identify index of QT grid (gQT) for each line
        self._QTmask = self.make_QTmask(self._ielem, self._iion)

        # Line strength: input shoud be ndarray not jnp array
        self.Sij0 = atomll.Sij0(self._A, self._gupper, self.nu_lines,
                                self._elower, self.QTref_284, self._QTmask,
                                Irwin)  # 211013

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
            list(
                map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 4],
                    self.ielem)))
        self.atomicmass = jnp.array(
            list(
                map(lambda x: ipccd[ipccd['ielem'] == x].iat[0, 5],
                    self.ielem)))
        df_ionE = atomllapi.load_ionization_energies()
        self.ionE = jnp.array(
            list(
                map(atomllapi.pick_ionE, self.ielem, self.iion, [
                    df_ionE,
                ] * len(self.ielem))))

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
        atomspecies_Roman = atomspecies.split(' ')[0] + '_' + 'I' * int(
            atomspecies.split(' ')[-1])
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
        return self.QT_interp(atomspecies, T) / self.QT_interp(
            atomspecies, Tref)

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
        return self.QT_interp_Irwin_Fe(
            T, atomspecies) / self.QT_interp_Irwin_Fe(Tref, atomspecies)

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
        listofQT = list(
            map(lambda x: jnp.interp(T, self.T_gQT, x),
                listofDA_gQT_eachspecies))
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
            sp_Roman = atomllapi.PeriodicTable[ielem] + '_' + 'I' * iion
            QTmask = np.where(self.pfdat['T[K]'] == sp_Roman)[0][0]
            return QTmask

        QTmask_sp = np.array(list(map(species_to_QTmask, ielem,
                                      iion))).astype('int')
        return QTmask_sp
