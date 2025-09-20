import pathlib
import warnings
import numpy as np
from radis.db.classes import get_molecule
from radis.levels.partfunc import PartFuncTIPS

from exojax.database.core.line_strength import line_strength_numpy
from exojax.database._common.hitranapi import molecid_hitran
from exojax.database._common.hitranapi import make_partition_function_grid_hitran
from exojax.utils.constants import Tref_original
from exojax.utils.isotopes import molmass_hitran
from exojax.database._common.setradis import _set_engine
from exojax.database._common.isotope_functions import _isotope_index_from_isotope_number
from exojax.database._common.partition_function import _QT_interp, _qr_interp, _qr_interp_lines

class MdbCommonHitempHitran:
    def __init__(
        self,
        path="CO",
        nurange=[-np.inf, np.inf],
        crit=0.0,
        elower_max=None,
        Ttyp=1000.0,
        isotope=1,
        gpu_transfer=False,
        activation=True,
        with_error=False,
        engine=None,
    ):
        """Molecular database for HITRAN/HITEMP form.

        Args:
            molecule: molecule
            nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
            crit: line strength lower limit for extraction
            elower_max: maximum lower state energy, Elower (cm-1)
            Ttyp: typical temperature to calculate Sij(T) used in crit
            isotope: isotope number, 0 or None = use all isotopes.
            gpu_transfer: tranfer data to jnp.array?
            activation: if True, the activation of mdb will be done when initialization, if False, the activation won't be done and it makes self.df attribute available.
            with_error: if True, uncertainty indices become available.
            engine: engine for radis api ("pytables" or "vaex" or None). if None, radis automatically determines. default to None

        """

        self.path = pathlib.Path(path).expanduser()
        self.molecid = molecid_hitran(str(self.path.stem))
        self.simple_molecule_name = get_molecule(self.molecid)
        self.crit = crit
        self.elower_max = elower_max
        self.Ttyp = Ttyp
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.isotope = isotope
        self.set_molmass()
        self.gpu_transfer = gpu_transfer
        self.activation = activation
        self.load_wavenum_min, self.load_wavenum_max = self.set_wavenum(nurange)
        self.with_error = with_error
        self.engine = _set_engine(engine)

    def QT_for_select_line(self, Ttyp):
        if self.isotope is None or self.isotope == 0:
            isotope_for_Qt = 1  # we use isotope=1 for QT
        else:
            isotope_for_Qt = int(self.isotope)
        Q = PartFuncTIPS(self.molecid, isotope_for_Qt)
        QTref = Q.at(T=Tref_original)
        QTtyp = Q.at(T=Ttyp)
        return QTref, QTtyp

    def set_wavenum(self, nurange):
        if nurange is None:
            wavenum_min = 0.0
            wavenum_max = 0.0
            self.activation = False
            warnings.warn("nurange=None. Nonactive mode.", UserWarning)
        else:
            wavenum_min = np.min(nurange)
            wavenum_max = np.max(nurange)
        if wavenum_min == -np.inf:
            wavenum_min = None
        if wavenum_max == np.inf:
            wavenum_max = None
        return wavenum_min, wavenum_max

    def activate(self, df, mask=None):
        """activation of moldb,

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
        self.gQT, self.T_gQT = make_partition_function_grid_hitran(
            self.molecid, self.uniqiso
        )

        if self.gpu_transfer:
            self.generate_jnp_arrays()

    def set_molmass(self):
        molmass_isotope, abundance_isotope = molmass_hitran()
        if self.isotope is None:
            self.molmass = molmass_isotope[self.simple_molecule_name][0]
        else:
            self.molmass = molmass_isotope[self.simple_molecule_name][self.isotope]

    def compute_load_mask(self, df, qrtyp):
        # wavelength
        mask = (df.wav > self.load_wavenum_min) & (df.wav < self.load_wavenum_max)
        mask = mask & (
            line_strength_numpy(self.Ttyp, df.int, df.wav, df.El, qrtyp) > self.crit
        )
        if self.elower_max is not None:
            mask = mask & (df.El < self.elower_max)
        return mask

    def apply_mask_mdb(self, mask):
        """apply mask for mdb class

        Args:
            mask: mask to be applied

        Examples:
            >>> mdb = MdbHitemp(emf, nus)
            >>> # we would extract the lines with n_air > 0.01
            >>> mask = mdb.n_air > 0.01
            >>> mdb.apply_mask_mdb(mask)
        """
        self.nu_lines = self.nu_lines[mask]
        self.line_strength_ref_original = self.line_strength_ref_original[mask]
        self.delta_air = self.delta_air[mask]
        self.A = self.A[mask]
        self.n_air = self.n_air[mask]
        self.gamma_air = self.gamma_air[mask]
        self.gamma_self = self.gamma_self[mask]
        self.elower = self.elower[mask]
        self.gpp = self.gpp[mask]
        # isotope
        self.isoid = self.isoid[mask]
        self.uniqiso = np.unique(self.isoid)
        if self.with_error:
            # uncertainties
            self.ierr = self.ierr[mask]

    def QT_interp(self, isotope, T):
        """interpolated partition function.

        Args:
            isotope: HITRAN isotope number starting from 1
            T: temperature

        Returns:
            Q(idx, T) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(isotope, self.uniqiso)
        return _QT_interp(isotope_index, T, self.T_gQT, self.gQT)

    def qr_interp(self, isotope, T, Tref):
        """interpolated partition function ratio.

        Args:
            isotope: HITRAN isotope number starting from 1
            T: temperature
            Tref: reference temperature

        Returns:
            qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
        """
        isotope_index = _isotope_index_from_isotope_number(isotope, self.uniqiso)
        return _qr_interp(isotope_index, T, self.T_gQT, self.gQT, Tref)

    def qr_interp_lines(self, T, Tref):
        """Partition Function ratio using HAPI partition data.
        (This function works for JAX environment.)

        Args:
            T: temperature (K)
            Tref: reference temperature (K)

        Returns:
            Qr_line, partition function ratio array for lines [Nlines]

        Note:
            Nlines=len(self.nu_lines)
        """
        return _qr_interp_lines(T, self.isoid, self.uniqiso, self.T_gQT, self.gQT, Tref)

    def exact_isotope_name(self, isotope):
        """exact isotope name

        Args:
            isotope (int): isotope number starting from 1

        Returns:
            str: exact isotope name such as (12C)(16O)
        """
        from exojax.utils.molname import exact_molecule_name_from_isotope

        return exact_molecule_name_from_isotope(self.simple_molecule_name, isotope)

    def line_strength(self, T):
        """line strength at T

        Args:
            T (float): temperature

        Returns:
            float: line strength at T
        """
        if self.isotope is None or self.isotope == 0:
            msg1 = "Currently all isotope mode is not fully compatible to MdbCommonHitempHitran."
            msg2 = "QT assumed isotope=1 instead."
            warnings.warn(msg1 + msg2, UserWarning)
            qr = self.qr_interp(1, T, Tref_original)
        else:
            qr = self.qr_interp(self.isotope, T, Tref_original)

        return line_strength_numpy(
            T,
            self.line_strength_ref_original,
            self.nu_lines,
            self.elower,
            qr,
            Tref_original,
        )

    def check_line_existence_in_nurange(self, df_load_mask):
        if len(df_load_mask) == 0:
            raise ValueError("No line found in ", self.nurange, "cm-1")

    def add_error(self):
        """uncertainty codes of HITRAN or HITEMP database
        ref.: Table 2 and 5 in HITRAN 2004 (Rothman et al. 2005)

        Returns:
            Uncertainty indices for 6 critical parameters
        """
        is_place = (
            lambda x, i: (x // 10 ** (i)) % 10
        )  # extract the digits for 10**i place (0<=i<=5)
        self.nu_lines_err = is_place(self.ierr, 5)  # 0-9
        self.line_strength_ref_err = is_place(self.ierr, 4)  # 0-8
        self.gamma_air_err = is_place(self.ierr, 3)  # 0-8
        self.gamma_self_err = is_place(self.ierr, 2)  # 0-8
        self.n_air_err = is_place(self.ierr, 1)
        self.delta_air_err = is_place(self.ierr, 0)  # 0-9
