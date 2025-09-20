"""Molecular database API class using a common API w/ RADIS = (CAPI)

* MdbExomol is the MDB for ExoMol
* MdbHitran is the MDB for HITRAN
* MdbHitemp is the MDB for HITEMP
* MdbCommonHitempHitran is the common MDB for HITEMP and HITRAN

Notes:
    If you use vaex as radis engine, hdf5 files are saved while pytables uses .h5 files.

"""

import pathlib
import warnings

import jax.numpy as jnp
import numpy as np
from packaging import version
from radis import __version__ as radis_version
from radis.api.exomolapi import (
    MdbExomol as CapiMdbExomol,
)  # MdbExomol in the common API
from exojax.database.core.broadening import gamma_natural as gn
from exojax.database.core.line_strength import line_strength_numpy
from exojax.database.molinfo import isotope_molmass
from exojax.utils.constants import Tref_original
from exojax.utils.molname import e2s
from exojax.database._common.setradis import _set_engine
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot

__all__ = ["MdbExomol"]


class MdbExomol(CapiMdbExomol):
    """molecular database of ExoMol form.

    MdbExomol is a class for ExoMol database. It inherits the CAPI class MdbExomol and adds some additional features.

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

    def __init__(
        self,
        path,
        nurange=[-np.inf, np.inf],
        crit=0.0,
        elower_max=None,
        Ttyp=1000.0,
        bkgdatm="H2",
        broadf=True,
        broadf_download=True,
        gpu_transfer=True,
        inherit_dataframe=False,
        optional_quantum_states=False,
        activation=True,
        local_databases="./",
        engine=None,
    ):
        """Molecular database for Exomol form.

        Args:
            path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
            nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid, if None, it starts as the nonactive mode
            crit: line strength lower limit for extraction
            Ttyp: typical temperature to calculate Sij(T) used in crit
            bkgdatm: background atmosphere for broadening (or broadener species in radis). e.g. H2, He, air
            broadf: if False, the default broadening parameters in .def file is used
            broadf_download: if False, not try to download the potential broadening files. default to True
            gpu_transfer: if True, some attributes will be transfered to jnp.array. False is recommended for PreMODIT.
            inherit_dataframe: if True, it makes self.df attribute available, which needs more DRAM when pickling.
            optional_quantum_states: if True, all of the fields available in self.df will be loaded. if False, the mandatory fields (i,E,g,J) will be loaded.
            activation: if True, the activation of mdb will be done when initialization, if False, the activation won't be done and it makes self.df attribute available.
            engine: engine for radis api ("pytables" or "vaex" or None). if None, radis automatically determines. default to None

        Note:
            The trans/states files can be very large. For the first time to read it, we convert it to HDF/vaex. After the second-time, we use the HDF5 format with vaex instead.
        """
        self.dbtype = "exomol"
        self.path = pathlib.Path(path).expanduser()
        self.exact_molecule_name = self.path.parents[0].stem
        self.database = str(self.path.stem)
        self.bkgdatm = bkgdatm
        # molecbroad = self.exact_molecule_name + '__' + self.bkgdatm
        self.gpu_transfer = gpu_transfer
        self.Ttyp = Ttyp
        self.broadf = broadf
        if radis_version >= "0.16":
            self.broadf_download = broadf_download
        else:
            print("radis==", radis_version)
            msg = "The current version of radis does not support broadf_download (requires >=0.16)."
            warnings.warn(msg, UserWarning)
        self.simple_molecule_name = e2s(self.exact_molecule_name)
        self.molmass = isotope_molmass(self.exact_molecule_name)
        self.skip_optional_data = not optional_quantum_states
        self.activation = activation
        wavenum_min, wavenum_max = self.set_wavenum(nurange)
        self.engine = _set_engine(engine)

        if radis_version >= "0.16":
            super().__init__(
                str(self.path),
                local_databases=local_databases,
                molecule=self.simple_molecule_name,
                name="EXOMOL-{molecule}",
                nurange=[wavenum_min, wavenum_max],
                engine=self.engine,
                crit=crit,
                broadf=self.broadf,
                broadf_download=self.broadf_download,
                cache=True,
                skip_optional_data=self.skip_optional_data,
            )
        else:
            super().__init__(
                str(self.path),
                local_databases=local_databases,
                molecule=self.simple_molecule_name,
                name="EXOMOL-{molecule}",
                nurange=[wavenum_min, wavenum_max],
                engine=self.engine,
                crit=crit,
                bkgdatm=self.bkgdatm,  # uses radis <= 0.15.2
                broadf=self.broadf,
                cache=True,
                skip_optional_data=self.skip_optional_data,
            )

        self.crit = crit
        self.elower_max = elower_max
        self.QTtyp = np.array(self.QT_interp(self.Ttyp))

        # Get cache files to load :
        mgr = self.get_datafile_manager()
        local_files = [mgr.cache_file(f) for f in self.trans_file]

        # data frame attribute:
        df = self.load(
            local_files,
            # lower_bound=([("Sij0", 0.0)]),
            output=self.engine,
        )

        self.df_load_mask = self.compute_load_mask(df)

        if self.activation:
            self.activate(df)
        if inherit_dataframe or not self.activation:
            print("DataFrame (self.df) available.")
            self.df = df

    def __eq__(self, other):
        """eq method for MdbExomol, definied by comparing all the attributes and important status

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, MdbExomol):
            return False

        eq_attributes = (
            all(self.A == other.A)
            and all(self.nu_lines == other.nu_lines)
            and all(self.elower == other.elower)
            and all(self.jlower == other.jlower)
            and all(self.jupper == other.jupper)
            and all(self.line_strength_ref_original == other.line_strength_ref_original)
            and all(self.logsij0 == other.logsij0)
            and all(self.gpp == other.gpp)
            and self.bkgdatm == other.bkgdatm
            and self.gpu_transfer == other.gpu_transfer
            and self.Ttyp == other.Ttyp
            and self.broadf == other.broadf
            and self.exact_molecule_name == other.exact_molecule_name
        )

        return eq_attributes

    def __ne__(self, other):
        return not self.__eq__(other)

    def attributes_from_dataframes(self, df_masked):
        """Generates attributes from (usually masked) data frame for Exomol

        Args:
            df_masked (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """

        if len(df_masked) == 0:
            raise ValueError("No line found in ", self.nurange, "cm-1")

        self._attributes_from_dataframes(df_masked)

    def _attributes_from_dataframes(self, df_masked):
        self.A = df_masked.A.values
        self.nu_lines = df_masked.nu_lines.values
        self.elower = df_masked.elower.values
        self.jlower = df_masked.jlower.values
        self.jupper = df_masked.jupper.values
        self.line_strength_ref_original = df_masked.Sij0.values
        self.logsij0 = np.log(self.line_strength_ref_original)
        self.gpp = df_masked.gup.values

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
        """Activates of moldb for Exomol,  including making attributes, computing broadening parameters, natural width, and transfering attributes to gpu arrays when self.gpu_transfer = True

        Notes:
            activation includes, making attributes, computing broadening parameters, natural width,
            and transfering attributes to gpu arrays when self.gpu_transfer = True

        Args:
            df: DataFrame
            mask: mask of DataFrame to be used for the activation, if None, no additional mask is applied.

        Note:
            self.df_load_mask is always applied when the activation.

        Examples:

            >>> # we would extract the line with delta nu = 2 here
            >>> mdb = MdbExomol(emf, nus, optional_quantum_states=True, activation=False)
            >>> load_mask = (mdb.df["v_u"] - mdb.df["v_l"] == 2)
            >>> mdb.activate(mdb.df, load_mask)


        """
        if mask is not None:
            mask = mask * self.df_load_mask
        else:
            mask = self.df_load_mask

        self.attributes_from_dataframes(df[mask])

        if version.parse(radis_version) <= version.parse("0.14"):
            self.compute_broadening(self.jlower.astype(int), self.jupper.astype(int))
        elif version.parse(radis_version) <= version.parse("0.15.2"):
            print("Broadener: ", self.bkgdatm)
            self.set_broadening_coef(df[mask], add_columns=False)
        else:
            # new broadener see radis#716, radis#742
            print("Broadener: ", self.bkgdatm)
            self.set_broadening_coef(df[mask], add_columns=False, species=self.bkgdatm)

        self.gamma_natural = gn(self.A)
        if self.gpu_transfer:
            self.generate_jnp_arrays()

    def compute_load_mask(self, df):
        # wavelength
        mask = (df.nu_lines > self.nurange[0]) & (df.nu_lines < self.nurange[1])
        QTtyp = np.array(self.QT_interp_numpy(self.Ttyp))
        QTref_original = np.array(self.QT_interp_numpy(Tref_original))
        mask = mask & (
            line_strength_numpy(
                self.Ttyp, df.Sij0, df.nu_lines, df.elower, QTtyp / QTref_original
            )
            > self.crit
        )
        if self.elower_max is not None:
            mask = mask & (df.elower < self.elower_max)
        return mask

    def apply_mask_mdb(self, mask):
        """Applys mask for mdb class for Exomol

        Args:
            mask: mask to be applied

        Examples:
            >>> mdb = MdbExomol(emf, nus)
            >>> # we would extract the lines with elower > 100.
            >>> mask = mdb.elower > 100.
            >>> mdb.apply_mask_mdb(mask)
        """
        self.A = self.A[mask]
        self.logsij0 = self.logsij0[mask]
        self.nu_lines = self.nu_lines[mask]
        self.dev_nu_lines = self.dev_nu_lines[mask]
        self.gamma_natural = self.gamma_natural[mask]
        self.alpha_ref = self.alpha_ref[mask]
        self.n_Texp = self.n_Texp[mask]
        self.elower = self.elower[mask]
        self.jlower = self.jlower[mask]
        self.jupper = self.jupper[mask]
        self.line_strength_ref_original = self.line_strength_ref_original[mask]
        self.gpp = self.gpp[mask]

    def generate_jnp_arrays(self):
        """(re)generates jnp.arrays.

        Note:
            We have nd arrays and jnp arrays. We usually apply the mask to nd arrays and then generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.
            logsij0 is computed assuming Tref=Tref_original because it is not used for PreMODIT.
        """
        # jnp arrays
        self.dev_nu_lines = jnp.array(self.nu_lines)
        self.logsij0 = jnp.array(np.log(self.line_strength_ref_original))

    def QT_interp(self, T):
        """interpolated partition function.

        Args:
            T: temperature

        Returns:
            Q(T) interpolated in jnp.array
        """
        return jnp.interp(T, self.T_gQT, self.gQT)

    def QT_interp_numpy(self, T):
        """interpolated partition function using numpy.

        Args:
            T: temperature

        Returns:
            Q(T) interpolated in np.array
        """
        return np.interp(T, self.T_gQT, self.gQT)

    def qr_interp(self, T, Tref):
        """interpolated partition function ratio.

        Args:
            T: temperature
            Tref: reference temperature

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated
        """
        return self.QT_interp(T) / self.QT_interp(Tref)

    def qr_interp_numpy(self, T, Tref):
        """interpolated partition function ratio numpy version.

        Args:
            T: temperature
            Tref: reference temperature

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated
        """
        return self.QT_interp_numpy(T) / self.QT_interp_numpy(Tref)

    def line_strength(self, T):
        """line strength at T

        Args:
            T (float): temperature

        Returns:
            float: line strength at T
        """
        qr = self.qr_interp_numpy(T, Tref_original)
        return line_strength_numpy(
            T,
            self.line_strength_ref_original,
            self.nu_lines,
            self.elower,
            qr,
            Tref_original,
        )

    # --- Snapshots / DTO export ---
    def to_snapshot(self) -> MDBSnapshot:
        """Export a data-only snapshot of this ExoMol MDB.

        The snapshot is immutable and contains only NumPy arrays and
        primitives, suitable for passing into opacity code without
        depending on this concrete database class.
        """
        meta = MDBMeta(
            dbtype="exomol",
            molmass=float(self.molmass),
            T_gQT=np.asarray(self.T_gQT),
            gQT=np.asarray(self.gQT),
        )

        lines = Lines(
            nu_lines=np.asarray(self.nu_lines),
            elower=np.asarray(self.elower),
            line_strength_ref_original=np.asarray(self.line_strength_ref_original),
        )

        n_Texp = (
            np.asarray(self.n_Texp) if hasattr(self, "n_Texp") and self.n_Texp is not None else None
        )
        alpha_ref = (
            np.asarray(self.alpha_ref) if hasattr(self, "alpha_ref") and self.alpha_ref is not None else None
        )

        return MDBSnapshot(
            meta=meta,
            lines=lines,
            n_Texp=n_Texp,
            alpha_ref=alpha_ref,
        )
