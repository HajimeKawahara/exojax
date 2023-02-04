"""Molecular database (MDB) class using a common API w/ RADIS = (CAPI), will be renamed.

* MdbExomol is the MDB for ExoMol
* MdbHit is the MDB for HITRAN or HITEMP
"""
from os.path import exists
import numpy as np
import jax.numpy as jnp
import pathlib
import vaex
import warnings
from exojax.spec.hitran import line_strength_numpy
from exojax.spec.hitran import gamma_natural as gn
from exojax.utils.constants import Tref_original
from exojax.utils.molname import e2s
from exojax.spec import hitranapi
from exojax.spec.hitranapi import molecid_hitran
from exojax.spec.molinfo import isotope_molmass
from exojax.utils.isotopes import molmass_hitran

from radis.api.exomolapi import MdbExomol as CapiMdbExomol  #MdbExomol in the common API
from radis.api.hitempapi import HITEMPDatabaseManager
from radis.api.hitranapi import HITRANDatabaseManager
from radis.api.hdf5 import update_pytables_to_vaex
from radis.db.classes import get_molecule
from radis.levels.partfunc import PartFuncTIPS
import warnings

__all__ = ['MdbExomol', 'MdbHitemp', 'MdbHitran']


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
        "line_strength_ref",
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
                 elower_max=None,
                 Ttyp=1000.,
                 bkgdatm='H2',
                 broadf=True,
                 gpu_transfer=True,
                 inherit_dataframe=False,
                 optional_quantum_states=False,
                 activation=True,
                 local_databases="./"):
        """Molecular database for Exomol form.

        Args:
            path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
            nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid, if None, it starts as the nonactive mode
            margin: margin for nurange (cm-1)
            crit: line strength lower limit for extraction
            Ttyp: typical temperature to calculate Sij(T) used in crit
            bkgdatm: background atmosphere for broadening. e.g. H2, He,
            broadf: if False, the default broadening parameters in .def file is used
            gpu_transfer: if True, some instances will be transfered to jnp.array. False is recommended for PreMODIT.
            inherit_dataframe: if True, it makes self.df instance available, which needs more DRAM when pickling.
            optional_quantum_states: if True, all of the fields available in self.df will be loaded. if False, the mandatory fields (i,E,g,J) will be loaded.
            activation: if True, the activation of mdb will be done when initialization, if False, the activation won't be done and it makes self.df instance available. 


        Note:
            The trans/states files can be very large. For the first time to read it, we convert it to HDF/vaex. After the second-time, we use the HDF5 format with vaex instead.
        """
        self.dbtype = "exomol"
        self.path = pathlib.Path(path).expanduser()
        self.exact_molecule_name = self.path.parents[0].stem
        self.database = str(self.path.stem)
        self.bkgdatm = bkgdatm
        #molecbroad = self.exact_molecule_name + '__' + self.bkgdatm
        self.Tref = Tref_original
        self.gpu_transfer = gpu_transfer
        self.Ttyp = Ttyp
        self.broadf = broadf
        self.simple_molecule_name = e2s(self.exact_molecule_name)
        self.molmass = isotope_molmass(self.exact_molecule_name)
        self.skip_optional_data = not optional_quantum_states
        self.activation = activation
        wavenum_min, wavenum_max = self.set_wavenum(nurange)

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
                         skip_optional_data=self.skip_optional_data)

        self.crit = crit
        self.elower_max = elower_max
        self.QTtyp = np.array(self.QT_interp(self.Ttyp))

        # Get cache files to load :
        mgr = self.get_datafile_manager()
        local_files = [mgr.cache_file(f) for f in self.trans_file]
        # data frame instance:
        df = self.load(
            local_files,
            columns=[k for k in self.__slots__ if k not in ["logsij0"]],
            lower_bound=([("Sij0", 0.0)]),
            output="vaex")

        self.df_load_mask = self.compute_load_mask(df)

        if self.activation:
            self.activate(df)
        if inherit_dataframe or not self.activation:
            print("DataFrame (self.df) available.")
            self.df = df
        
    def set_wavenum(self, nurange):
        if nurange is None:
            wavenum_min = 0.0
            wavenum_max = 0.0
            self.activation = False
            warnings.warn("nurange=None. Nonactive mode.", UserWarning)
        else:
            wavenum_min, wavenum_max = np.min(nurange), np.max(nurange)
        if wavenum_min == -np.inf:
            wavenum_min = None
        if wavenum_max == np.inf:
            wavenum_max = None
        return wavenum_min, wavenum_max

    def activate(self, df, mask=None):
        """activation of moldb, 
        
        Notes:
            activation includes, making instances, computing broadening parameters, natural width, 
            and transfering instances to gpu arrays when self.gpu_transfer = True

        Args:
            df: DataFrame
            mask: mask of DataFrame to be used for the activation, if None, no additional mask is applied.

        Note:
            self.df_load_mask is always applied when the activation.

        Examples:
            
            >>> # we would extract the line with delta nu = 2 here
            >>> mdb = api.MdbExomol(emf, nus, optional_quantum_states=True, activation=False)
            >>> load_mask = (mdb.df["v_u"] - mdb.df["v_l"] == 2)
            >>> mdb.activate(mdb.df, load_mask)


        """
        if mask is not None:
            mask = mask * self.df_load_mask
        else:
            mask = self.df_load_mask

        self.instances_from_dataframes(df[mask])
        self.compute_broadening(self.jlower, self.jupper)
        self.gamma_natural = gn(self.A)
        if self.gpu_transfer:
            self.generate_jnp_arrays()

    def compute_load_mask(self, df):

        #wavelength
        mask = (df.nu_lines > self.nurange[0]-self.margin) \
                    * (df.nu_lines < self.nurange[1]+self.margin)
        QTtyp = np.array(self.QT_interp(self.Ttyp))
        QTref_original = np.array(self.QT_interp(Tref_original))
        mask *= (line_strength_numpy(self.Ttyp, df.Sij0, df.nu_lines,
                                     df.elower, QTtyp / QTref_original) >
                 self.crit)
        if self.elower_max is not None:
            mask *= (df.elower < self.elower_max)
        return mask

    def instances_from_dataframes(self, df_masked):
        """generate instances from (usually masked) data frame

        Args:
            df_masked (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """
        if isinstance(df_masked, vaex.dataframe.DataFrameLocal):
            self.A = df_masked.A.values
            self.nu_lines = df_masked.nu_lines.values
            self.elower = df_masked.elower.values
            self.jlower = df_masked.jlower.values
            self.jupper = df_masked.jupper.values
            self.line_strength_ref = df_masked.Sij0.values
            self.gpp = df_masked.gup.values
        else:
            raise ValueError("Use vaex dataframe as input.")

    def Sij0(self):
        """Deprecated line_strength_ref. 

        Returns:
            ndarray: line_strength_ref
        """
        msg = "Sij0 instance was replaced to line_strength_ref and will be removed."
        warnings.warn(msg, DeprecationWarning)
        return self.line_strength_ref

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        # jnp arrays
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.line_strength_ref))
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
        return self.QT_interp(T) / self.QT_interp(self.Tref)

    def change_reference_temperature(self, Tref_new):
        """change the reference temperature Tref and recompute Sij0

        Args:
            Tref_new (float): new Tref in Kelvin
        """
        print("Tref changed: " + str(self.Tref) + "K->" + str(Tref_new) + "K")
        qr = self.qr_interp(Tref_new)
        self.line_strength_ref = line_strength_numpy(Tref_new,
                                                     self.line_strength_ref,
                                                     self.nu_lines,
                                                     self.elower, qr,
                                                     self.Tref)
        self.Tref = Tref_new


class MdbHitemp(HITEMPDatabaseManager):
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
                 elower_max=None,
                 Ttyp=1000.,
                 isotope=1,
                 gpu_transfer=False,
                 inherit_dataframe=False):
        """Molecular database for HITRAN/HITEMP form.

        Args:
           path: path for HITEMP par file
           nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           elower_max: maximum lower state energy, Elower (cm-1)
           Ttyp: typical temperature to calculate Sij(T) used in crit
           isotope: isotope number, 0 or None = use all isotopes. 
           gpu_transfer: tranfer data to jnp.array?
           inherit_dataframe: if True, it makes self.df instance available, which needs more DRAM when pickling.
        """

        self.dbtype = "hitran"
        self.path = pathlib.Path(path).expanduser()
        self.molecid = molecid_hitran(str(self.path.stem))
        self.simple_molecule_name = get_molecule(self.molecid)
        self.crit = crit
        self.elower_max = elower_max
        self.Tref = Tref_original
        self.Ttyp = Ttyp
        self.margin = margin
        self.nurange = [np.min(nurange), np.max(nurange)]
        load_wavenum_min = self.nurange[0] - self.margin
        load_wavenum_max = self.nurange[1] + self.margin
        self.isotope = isotope
        self.set_molmass()

        super().__init__(
            molecule=self.simple_molecule_name,
            name="HITEMP-{molecule}",
            local_databases=self.path.parent,
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
        # if not self.is_registered():
        #    self.register()

        clean_cache_files = True
        if len(download_files) > 0 and clean_cache_files:
            self.clean_download_files()

        # Load and return
        files_loaded = self.keep_only_relevant(local_files, load_wavenum_min,
                                               load_wavenum_max)
        columns = None,
        output = "vaex"

        isotope_dfform = _convert_proper_isotope(self.isotope)
        df = self.load(
            files_loaded,  # filter other files,
            columns=columns,
            within=[("iso",
                     isotope_dfform)] if isotope_dfform is not None else [],
            # for relevant files, get only the right range :
            lower_bound=[("wav", load_wavenum_min)]
            if self.nurange[0] is not None else [],
            upper_bound=[("wav", load_wavenum_max)]
            if self.nurange[1] is not None else [],
            output=output,
        )

        self.isoid = df.iso
        self.uniqiso = np.unique(df.iso.values)
        for iso in self.uniqiso:
            Q = PartFuncTIPS(self.molecid, iso)
            QTref = Q.at(T=self.Tref)
            QTtyp = Q.at(T=self.Ttyp)
            load_mask = self.compute_load_mask(df, QTtyp / QTref)
        self.instances_from_dataframes(df[load_mask])
        self.gQT, self.T_gQT = hitranapi.make_partition_function_grid_hitran(
            self.molecid, self.uniqiso)

        if gpu_transfer:
            self.generate_jnp_arrays()

        if inherit_dataframe:
            self.df = df

    def set_molmass(self):
        molmass_isotope, abundance_isotope = molmass_hitran()
        if self.isotope is None:
            self.molmass = molmass_isotope[self.simple_molecule_name][0]
        else:
            self.molmass = molmass_isotope[self.simple_molecule_name][
                self.isotope]

    def compute_load_mask(self, df, qrtyp):
        #wavelength
        mask = (df.wav > self.nurange[0]-self.margin) \
                    * (df.wav < self.nurange[1]+self.margin)
        mask *= (line_strength_numpy(self.Ttyp, df.int, df.wav, df.El, qrtyp) >
                 self.crit)
        if self.elower_max is not None:
            mask *= (df.elower < self.elower_max)
        return mask

    def instances_from_dataframes(self, df_masked):
        """generate instances from (usually masked) data farame

        Args:
            df_load_mask (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """
        if isinstance(df_masked, vaex.dataframe.DataFrameLocal):
            self.nu_lines = df_masked.wav.values
            self.line_strength_ref = df_masked.int.values
            self.delta_air = df_masked.Pshft.values
            self.A = df_masked.A.values
            self.n_air = df_masked.Tdpair.values
            self.gamma_air = df_masked.airbrd.values
            self.gamma_self = df_masked.selbrd.values
            self.elower = df_masked.El.values
            self.gpp = df_masked.gp.values
            #isotope
            self.isoid = df_masked.iso.values
            self.uniqiso = np.unique(self.isoid)
        else:
            raise ValueError("Use vaex dataframe as input.")

    def Sij0(self):
        """Deprecated line_strength_ref. 

        Returns:
            ndarray: line_strength_ref
        """
        msg = "Sij0 instance was replaced to line_strength_ref and will be removed."
        warnings.warn(msg, DeprecationWarning)
        return self.line_strength_ref

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.
        
        Note:
           We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.
        
        """
        # jnp.array copy from the copy sources
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.line_strength_ref))
        self.line_strength_ref = jnp.array(self.line_strength_ref)
        self.delta_air = jnp.array(self.delta_air)
        self.A = jnp.array(self.A)
        self.n_air = jnp.array(self.n_air)
        self.gamma_air = jnp.array(self.gamma_air)
        self.gamma_self = jnp.array(self.gamma_self)
        self.elower = jnp.array(self.elower)
        self.gpp = jnp.array(self.gpp)

    def QT_interp(self, isotope, T):
        """interpolated partition function.

        Args:
           isotope: HITRAN isotope number starting from 1
           T: temperature

        Returns:
           Q(idx, T) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(
            isotope, self.uniqiso)
        return _QT_interp(isotope_index, T, self.T_gQT, self.gQT)

    def qr_interp(self, isotope, T):
        """interpolated partition function ratio.

        Args:
            isotope: HITRAN isotope number starting from 1
            T: temperature

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(
            isotope, self.uniqiso)
        return _qr_interp(isotope_index, T, self.T_gQT, self.gQT, self.Tref)

    def qr_interp_lines(self, T):
        """Partition Function ratio using HAPI partition data.
        (This function works for JAX environment.)

        Args:
            T: temperature (K)

        Returns:
            Qr_line, partition function ratio array for lines [Nlines]

        Note:
            Nlines=len(self.nu_lines)
        """
        return _qr_interp_lines(T, self.isoid, self.uniqiso, self.T_gQT,
                                self.gQT, self.Tref)

    def exact_isotope_name(self, isotope):
        """exact isotope name

        Args:
            isotope (int): isotope number starting from 1

        Returns:
            str: exact isotope name such as (12C)(16O)
        """
        from exojax.utils.molname import exact_hitran_isotope_name_from_isotope
        return exact_hitran_isotope_name_from_isotope(
            self.simple_molecule_name, isotope)

    def change_reference_temperature(self, Tref_new):
        """change the reference temperature Tref and recompute Sij0

        Args:
            Tref_new (float): new Tref in Kelvin
        """
        print("Change the reference temperature from " + str(self.Tref) +
              "K to " + str(Tref_new) + " K.")
        if self.isotope is None or self.isotope == 0:
            msg1 = "Currently all isotope mode is not fully compatible to change_reference_temperature."
            msg2 = "QT used in change_reference_temperature is assumed isotope=1 instead."
            warnings.warn(msg1 + msg2, UserWarning)
            qr = self.qr_interp(1, Tref_new)
        else:
            qr = self.qr_interp(self.isotope, Tref_new)

        self.line_strength_ref = line_strength_numpy(Tref_new,
                                                     self.line_strength_ref,
                                                     self.nu_lines,
                                                     self.elower, qr,
                                                     self.Tref)
        self.Tref = Tref_new


MdbHit = MdbHitemp  #compatibility


class MdbHitran(HITRANDatabaseManager):
    """molecular database of HITRAN

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
                 elower_max=None,
                 Ttyp=1000.,
                 isotope=0,
                 gpu_transfer=False,
                 inherit_dataframe=False):
        """Molecular database for HITRAN/HITEMP form.

        Args:
           path: path for HITRAN/HITEMP par file
           nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           elower_max: maximum lower state energy, Elower (cm-1)
           Ttyp: typical temperature to calculate Sij(T) used in crit
           isotope: isotope number. 0 or None= use all isotopes. 
           gpu_transfer: tranfer data to jnp.array?
           inherit_dataframe: if True, it makes self.df instance available, which needs more DRAM when pickling.
        """
        self.dbtype = "hitran"
        self.path = pathlib.Path(path).expanduser()
        self.molecid = molecid_hitran(str(self.path.stem))
        self.simple_molecule_name = get_molecule(self.molecid)

        #numinf, numtag = hitranapi.read_path(self.path)
        self.crit = crit
        self.elower_max = elower_max
        self.Tref = Tref_original
        self.Ttyp = Ttyp
        self.margin = margin
        self.nurange = [np.min(nurange), np.max(nurange)]
        load_wavenum_min = self.nurange[0] - self.margin
        load_wavenum_max = self.nurange[1] + self.margin
        self.isotope = isotope
        self.set_molmass()

        super().__init__(
            molecule=self.simple_molecule_name,
            name="HITRAN-{molecule}",
            local_databases=self.path.parent,
            engine="default",
            verbose=True,
            parallel=True,
        )

        # Get list of all expected local files for this database:
        local_file = self.get_filenames()

        # Download files
        download_files = self.get_missing_files(local_file)
        if download_files:
            self.download_and_parse(download_files,
                                    cache=True,
                                    parse_quanta=True)

        # Register
        #if not self.is_registered():
        #    self.register()

        if len(download_files) > 0:
            self.clean_download_files()

        # Load and return
        columns = None
        output = "vaex"

        isotope_dfform = _convert_proper_isotope(self.isotope)
        df = self.load(
            local_file,
            columns=columns,
            within=[("iso",
                     isotope_dfform)] if isotope_dfform is not None else [],
            # for relevant files, get only the right range :
            lower_bound=[("wav", load_wavenum_min)]
            if load_wavenum_min is not None else [],
            upper_bound=[("wav", load_wavenum_max)]
            if load_wavenum_max is not None else [],
            output=output,
        )

        self.isoid = df.iso
        self.uniqiso = np.unique(df.iso.values)
        for iso in self.uniqiso:
            Q = PartFuncTIPS(self.molecid, iso)
            QTref = Q.at(T=self.Tref)
            QTtyp = Q.at(T=self.Ttyp)
            load_mask = self.compute_load_mask(df, QTtyp / QTref)
        self.instances_from_dataframes(df[load_mask])
        self.gQT, self.T_gQT = hitranapi.make_partition_function_grid_hitran(
            self.molecid, self.uniqiso)

        if gpu_transfer:
            self.generate_jnp_arrays()

        if inherit_dataframe:
            self.df = df

    def set_molmass(self):
        molmass_isotope, abundance_isotope = molmass_hitran()
        if self.isotope is None:
            self.molmass = molmass_isotope[self.simple_molecule_name][0]
        else:
            self.molmass = molmass_isotope[self.simple_molecule_name][
                self.isotope]

    def compute_load_mask(self, df, qrtyp):
        #wavelength
        mask = (df.wav > self.nurange[0]-self.margin) \
                    * (df.wav < self.nurange[1]+self.margin)
        mask *= (line_strength_numpy(self.Ttyp, df.int, df.wav, df.El, qrtyp) >
                 self.crit)
        if self.elower_max is not None:
            mask *= (df.elower < self.elower_max)
        return mask

    def instances_from_dataframes(self, df_load_mask):
        """generate instances from (usually masked) data farame

        Args:
            df_load_mask (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """
        if isinstance(df_load_mask, vaex.dataframe.DataFrameLocal):
            self.nu_lines = df_load_mask.wav.values
            self.line_strength_ref = df_load_mask.int.values
            self.delta_air = df_load_mask.Pshft.values
            self.isoid = df_load_mask.iso.values
            self.uniqiso = np.unique(self.isoid)
            self.A = df_load_mask.A.values
            self.n_air = df_load_mask.Tdpair.values
            self.gamma_air = df_load_mask.airbrd.values
            self.gamma_self = df_load_mask.selbrd.values
            self.elower = df_load_mask.El.values
            self.gpp = df_load_mask.gp.values
        else:
            raise ValueError("Use vaex dataframe as input.")

    def Sij0(self):
        """Deprecated line_strength_ref. 

        Returns:
            ndarray: line_strength_ref
        """
        msg = "Sij0 instance was replaced to line_strength_ref and will be removed."
        warnings.warn(msg, DeprecationWarning)
        return self.line_strength_ref

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.
        
        Note:
           We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.
        
        """
        # jnp.array copy from the copy sources
        self.nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.line_strength_ref))
        self.line_strength_ref = jnp.array(self.line_strength_ref)
        self.delta_air = jnp.array(self.delta_air)
        self.A = jnp.array(self.A)
        self.n_air = jnp.array(self.n_air)
        self.gamma_air = jnp.array(self.gamma_air)
        self.gamma_self = jnp.array(self.gamma_self)
        self.elower = jnp.array(self.elower)
        self.gpp = jnp.array(self.gpp)

    def QT_interp(self, isotope, T):
        """interpolated partition function.

        Args:
           isotope: HITRAN isotope number starting from 1
           T: temperature

        Returns:
           Q(idx, T) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(
            isotope, self.uniqiso)
        return _QT_interp(isotope_index, T, self.T_gQT, self.gQT)

    def qr_interp(self, isotope, T):
        """interpolated partition function ratio.

        Args:
            isotope: HITRAN isotope number starting from 1
            T: temperature

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(
            isotope, self.uniqiso)
        return _qr_interp(isotope_index, T, self.T_gQT, self.gQT, self.Tref)

    def qr_interp_lines(self, T):
        """Partition Function ratio using HAPI partition data.
        (This function works for JAX environment.)

        Args:
           T: temperature (K)

        Returns:
           Qr_line, partition function ratio array for lines [Nlines]

        Note:
           Nlines=len(self.nu_lines)
        """
        return _qr_interp_lines(T, self.isoid, self.uniqiso, self.T_gQT,
                                self.gQT, self.Tref)

    def exact_isotope_name(self, isotope):
        """exact isotope name

        Args:
            isotope (int): isotope number starting from 1

        Returns:
            str: exact isotope name such as (12C)(16O)
        """
        from exojax.utils.isotopes import exact_hitran_isotope_name_from_isotope
        return exact_hitran_isotope_name_from_isotope(
            self.simple_molecule_name, isotope)

    def change_reference_temperature(self, Tref_new):
        """change the reference temperature Tref and recompute Sij0

        Args:
            Tref_new (float): new Tref in Kelvin
        """
        print("Change the reference temperature from " + str(self.Tref) +
              "K to " + str(Tref_new) + " K.")
        qr = self.qr_interp(Tref_new)
        self.line_strength_ref = line_strength_numpy(Tref_new,
                                                     self.line_strength_ref,
                                                     self.nu_lines,
                                                     self.elower, qr,
                                                     self.Tref)
        self.Tref = Tref_new


def _convert_proper_isotope(isotope):
    """covert isotope (int) to proper type for df 

    Args:
        isotope (int or other type): isotope

    Returns:
        str: proper isotope type
    """
    if isotope == 0:
        return None
    elif isotope is not None and type(isotope) == int:
        return str(isotope)
    elif isotope is None:
        return isotope
    else:
        raise ValueError("Invalid isotope type")


def _isotope_index_from_isotope_number(isotope, uniqiso):
    """isotope index given HITRAN/HITEMP isotope number

        Args:
            isotope (int): isotope number
            uniqiso (nd int array): unique isotope array 

        Returns:
            int: isotope_index for T_gQT and gQT  
        """
    isotope_index = np.where(uniqiso == isotope)[0][0]
    return isotope_index


def _QT_interp(isotope_index, T, T_gQT, gQT):
    """interpolated partition function.

        Note:
            isotope_index is NOT isotope (number for HITRAN). 
            isotope_index is index for gQT and T_gQT.
            _isotope_index_from_isotope_number can be used 
            to get isotope index from isotope.
            
        Args:
            isotope index: isotope index, index from 0 to len(uniqiso) - 1
            T: temperature
            gQT: jnp array of partition function grid
            T_gQT: jnp array of temperature grid for gQT

        Returns:
            Q(idx, T) interpolated in jnp.array
        """

    return jnp.interp(T, T_gQT[isotope_index], gQT[isotope_index])


def _qr_interp(isotope_index, T, T_gQT, gQT, Tref):
    """interpolated partition function ratio.

        Note:
            isotope_index is NOT isotope (number for HITRAN). 
            isotope_index is index for gQT and T_gQT.
            _isotope_index_from_isotope_number can be used 
            to get isotope index from isotope.
    
        Args:
            isotope index: isotope index, index from 0 to len(uniqiso) - 1
            T: temperature
            gQT: jnp array of partition function grid
            T_gQT: jnp array of temperature grid for gQT
            Tref: reference temperature in K

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
    return _QT_interp(isotope_index, T, T_gQT, gQT) / _QT_interp(
        isotope_index, Tref, T_gQT, gQT)


def _qr_interp_lines(T, isoid, uniqiso, T_gQT, gQT, Tref):
    """Partition Function ratio using HAPI partition data.
        (This function works for JAX environment.)

        Args:
            T: temperature (K)
            isoid:
            uniqiso:
            gQT: jnp array of partition function grid
            T_gQT: jnp array of temperature grid for gQT
            Tref: reference temperature in K

        Returns:
            Qr_line, partition function ratio array for lines [Nlines]

        Note:
            Nlines=len(self.nu_lines)
        """
    qr_line = jnp.zeros(len(isoid))
    for isotope in uniqiso:
        mask_idx = np.where(isoid == isotope)
        isotope_index = _isotope_index_from_isotope_number(isotope, uniqiso)
        qr_each_isotope = _qr_interp(isotope_index, T, T_gQT, gQT, Tref)
        qr_line = qr_line.at[jnp.index_exp[mask_idx]].set(qr_each_isotope)
    return qr_line
