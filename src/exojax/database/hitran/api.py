import jax.numpy as jnp
import numpy as np
from radis.api.hitranapi import HITRANDatabaseManager
from exojax.database._common.commonapi import MdbCommonHitempHitran
from exojax.database._common.isotope_functions import _convert_proper_isotope
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot


class MdbHitran(MdbCommonHitempHitran, HITRANDatabaseManager):
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

    def __init__(
        self,
        molecule_path,
        nurange=[-np.inf, np.inf],
        crit=0.0,
        elower_max=None,
        Ttyp=1000.0,
        isotope=0,
        gpu_transfer=False,
        inherit_dataframe=False,
        activation=True,
        parfile=None,
        nonair_broadening=False,
        with_error=False,
        engine=None,
    ):
        """Molecular database for HITRAN/HITEMP form.

        Args:
            path: path for HITRAN/HITEMP par file
            nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
            crit: line strength lower limit for extraction
            elower_max: maximum lower state energy, Elower (cm-1)
            Ttyp: typical temperature to calculate Sij(T) used in crit
            isotope: isotope number. 0 or None= use all isotopes.
            gpu_transfer: tranfer data to jnp.array?
            inherit_dataframe: if True, it makes self.df attribute available, which needs more DRAM when pickling.
            activation: if True, the activation of mdb will be done when initialization, if False, the activation won't be done and it makes self.df attribute available.
            nonair_broadening: If True, background atmospheric broadening parameters(n and gamma) other than air will also be downloaded (e.g. h2, he...)
            with_error: if True, uncertainty indices become available. (Please set drop_non_numeric=False in radis.api.hitranapi)
            engine: engine for radis api ("pytables" or "vaex" or None). if None, radis automatically determines. default to None
        """
        self.dbtype = "hitran"
        MdbCommonHitempHitran.__init__(
            self,
            path=molecule_path,
            nurange=nurange,
            crit=crit,
            elower_max=elower_max,
            Ttyp=Ttyp,
            isotope=isotope,
            gpu_transfer=gpu_transfer,
            activation=activation,
            with_error=with_error,
            engine=engine,
        )

        # HITRAN ONLY FUNCTIONALITY
        if nonair_broadening:
            self.nonair_broadening = True
            extra_params = "all"
        else:
            self.nonair_broadening = False
            extra_params = None

        HITRANDatabaseManager.__init__(
            self,
            molecule=self.simple_molecule_name,
            name="HITRAN-{molecule}",
            local_databases=self.path.parent,
            engine=self.engine,
            verbose=True,
            parallel=True,
            extra_params=extra_params,
        )

        # Get list of all expected local files for this database:
        local_file = self.get_filenames()

        # Download files
        if with_error:
            # to distinguish hdf5 files with error and without error.
            local_file = [local_file[0].split(".hdf5")[0] + "_werr.hdf5"]
        download_files = self.get_missing_files(local_file)
        if download_files:
            self.download_and_parse(
                download_files,
                cache=True,
                parse_quanta=True,
                add_HITRAN_uncertainty_code=with_error,
            )

        if len(download_files) > 0:
            self.clean_download_files()

        # Load and return
        columns = None
        output = self.engine

        isotope_dfform = _convert_proper_isotope(self.isotope)
        df = self.load(
            local_file,
            columns=columns,
            within=[("iso", isotope_dfform)] if isotope_dfform is not None else [],
            # for relevant files, get only the right range :
            # lower_bound=[("wav", load_wavenum_min)]
            # if load_wavenum_min is not None else [],
            # upper_bound=[("wav", load_wavenum_max)]
            # if load_wavenum_max is not None else [],
            output=output,
        )

        self.isoid = df.iso
        self.uniqiso = np.unique(df.iso.values)
        QTref, QTtyp = self.QT_for_select_line(Ttyp)
        self.df_load_mask = self.compute_load_mask(df, QTtyp / QTref)
        if self.activation:
            self.activate(df)
        if inherit_dataframe or not self.activation:
            print("DataFrame (self.df) available.")
            self.df = df

    def attributes_from_dataframes(self, df_load_mask):
        """generate attributes from (usually masked) data farame

        Args:
            df_load_mask (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """
        self.check_line_existence_in_nurange(df_load_mask)
        self._attributes_from_dataframes_vaex(df_load_mask)

    def __eq__(self, other):
        """eq method for MdbHitran, definied by comparing all the attributes

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, MdbHitran):
            return False

        eq_attributes = (
            all(self.nu_lines == other.nu_lines)
            and all(self.line_strength_ref_original == other.line_strength_ref_original)
            and all(self.logsij0 == other.logsij0)
            and all(self.delta_air == other.delta_air)
            and all(self.A == other.A)
            and all(self.n_air == other.n_air)
            and all(self.gamma_air == other.gamma_air)
            and all(self.gamma_self == other.gamma_self)
            and all(self.elower == other.elower)
            and all(self.gpp == other.gpp)
            and all(self.isoid == other.isoid)
            and all(self.uniqiso == other.uniqiso)
            and self.isotope == other.isotope
            and self.simple_molecule_name == other.simple_molecule_name
            and self.gpu_transfer == other.gpu_transfer
            and self.Ttyp == other.Ttyp
            and self.elower_max == other.elower_max
        )
        eq_attributes = self._if_exist_check_eq_list(other, "ierr", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "n_h2", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "n_he", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "n_co2", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "n_h2o", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "gamma_h2", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "gamma_he", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "gamma_co2", eq_attributes)
        eq_attributes = self._if_exist_check_eq_list(other, "gamma_h2o", eq_attributes)

        return eq_attributes

    def _if_exist_check_eq_list(self, other, attribute, eq_attributes):
        if hasattr(self, attribute) and hasattr(other, attribute):
            return eq_attributes and all(
                getattr(self, attribute) == getattr(other, attribute)
            )
        elif not hasattr(self, attribute) and not hasattr(other, attribute):
            return eq_attributes
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _attributes_from_dataframes_vaex(self, df_load_mask):
        self.nu_lines = df_load_mask.wav.values
        self.line_strength_ref_original = df_load_mask.int.values
        self.logsij0 = np.log(self.line_strength_ref_original)
        self.delta_air = df_load_mask.Pshft.values
        self.A = df_load_mask.A.values
        self.n_air = df_load_mask.Tdpair.values
        self.gamma_air = df_load_mask.airbrd.values
        self.gamma_self = df_load_mask.selbrd.values
        self.elower = df_load_mask.El.values
        self.gpp = df_load_mask.gp.values
        # isotope
        self.isoid = df_load_mask.iso.values
        self.uniqiso = np.unique(self.isoid)
        if self.with_error:
            # uncertainties
            self.ierr = df_load_mask.ierr.values

        if hasattr(df_load_mask, "n_h2") and self.nonair_broadening:
            self.n_h2 = df_load_mask.n_h2.values
            self.gamma_h2 = df_load_mask.gamma_h2.values

        if hasattr(df_load_mask, "n_he") and self.nonair_broadening:
            self.n_he = df_load_mask.n_he.values
            self.gamma_he = df_load_mask.gamma_he.values

        if hasattr(df_load_mask, "n_co2") and self.nonair_broadening:
            self.n_co2 = df_load_mask.n_co2.values
            self.gamma_co2 = df_load_mask.gamma_co2.values

        if hasattr(df_load_mask, "n_h2o") and self.nonair_broadening:
            self.n_h2o = df_load_mask.n_h2o.values
            self.gamma_h2o = df_load_mask.gamma_h2o.values

    def generate_jnp_arrays(self):
        """(re)generate jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        # jnp.array copy from the copy sources
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.line_strength_ref_original))

    # --- Snapshots / DTO export ---
    def to_snapshot(self) -> MDBSnapshot:
        """Export a data-only snapshot for HITRAN-schema databases.

        Contains NumPy arrays only to decouple opacity code from DB classes.
        """
        meta = MDBMeta(
            dbtype="hitran",
            molmass=float(self.molmass),
            T_gQT=np.asarray(self.T_gQT),
            gQT=np.asarray(self.gQT),
        )

        lines = Lines(
            nu_lines=np.asarray(self.nu_lines),
            elower=np.asarray(self.elower),
            line_strength_ref_original=np.asarray(self.line_strength_ref_original),
        )

        return MDBSnapshot(
            meta=meta,
            lines=lines,
            isotope=np.asarray(self.isoid) if hasattr(self, "isoid") else None,
            uniqiso=np.asarray(self.uniqiso) if hasattr(self, "uniqiso") else None,
            n_air=np.asarray(self.n_air) if hasattr(self, "n_air") else None,
            gamma_air=np.asarray(self.gamma_air) if hasattr(self, "gamma_air") else None,
        )
