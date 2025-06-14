import warnings
import jax.numpy as jnp
import numpy as np

from exojax.opacity.base import OpaCalc
from exojax.opacity import initspec
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav
from exojax.utils.jaxstatus import check_jax64bit
from exojax.utils.instfunc import resolution_eslog


class OpaModit(OpaCalc):
    """Opacity Calculator Class for MODIT

    Attributes:
        opainfo: information set used in MODIT: cont_nu, index_nu, R, pmarray

    """

    def __init__(
        self,
        mdb,
        nu_grid,
        Tarr_list=None,
        Parr=None,
        Pself_ref=None,
        dit_grid_resolution=0.2,
        allow_32bit=False,
        alias="close",
        cutwing=1.0,
        wavelength_order="descending",
    ):
        """initialization of OpaModit

        Note:
            Tarr_list and Parr are used to compute xsmatrix. No need for xsvector

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            Tarr_list (1d or 2d array, optional): tempearture array to be tested such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr (1d array, optional): pressure array in bar
            Pself_ref (1d array, optional): self pressure array in bar. Defaults to None. If None Pself = 0.0.
            dit_grid_resolution (float, optional): dit grid resolution. Defaxults to 0.2.
            allow_32bit (bool, optional): If True, allow 32bit mode of JAX. Defaults to False.
            alias (str, optional): If "open", opa will give the open-type cross-section (with aliasing parts). Defaults to "close".
            cutwing (float, optional): wingcut for the convolution used in open cross section. Defaults to 1.0. For alias="close", always 1.0 is used by definition.
            wavlength order: wavelength order: "ascending" or "descending"

        Raises:
            ValueError: _description_
        """
        super().__init__(nu_grid)
        check_jax64bit(allow_32bit)

        # default setting
        self.method = "modit"
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.dit_grid_resolution = dit_grid_resolution
        if not self.mdb.gpu_transfer:
            raise ValueError("For MODIT, gpu_transfer should be True in mdb.")
        self.apply_params()
        if Tarr_list is not None and Parr is not None:
            self.setdgm(Tarr_list, Parr, Pself_ref=Pself_ref)
        else:
            warnings.warn("Tarr_list/Parr are needed for xsmatrix.", UserWarning)
        self.alias = alias
        self.cutwing = cutwing
        self.set_aliasing()

    def __eq__(self, other):
        """eq method for OpaModit, definied by comparing all the attributes and important status

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, OpaModit):
            return False

        eq_attributes = (
            (self.mdb == other.mdb)
            and (self.wavelength_order == other.wavelength_order)
            and all(self.nu_grid == other.nu_grid)
        )

        return eq_attributes

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_params(self):
        self.dbtype = self.mdb.dbtype
        self.opainfo = initspec.init_modit(self.mdb.nu_lines, self.nu_grid)
        self.ready = True

    def xsvector(self, T, P, Pself=0.0):
        """cross section vector

        Args:
            T (float): temperature
            P (float): pressure in bar
            Pself (float, optional): self pressure for HITEMP/HITRAN. Defaults to 0.0.

        Returns:
            1D array: cross section in cm2
        """
        from exojax.database.hitran import gamma_natural, normalized_doppler_sigma
        from exojax.database.exomol import gamma_exomol
        from exojax.database.hitran import gamma_hitran, line_strength
        from exojax.opacity.modit.modit import xsvector_open_zeroscan, xsvector_zeroscan
        from exojax.opacity._common.set_ditgrid import ditgrid_log_interval

        cont_nu, index_nu, R, pmarray = self.opainfo

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, Tref_original)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air, self.mdb.gamma_self
            ) + gamma_natural(self.mdb.A)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T, Tref_original)
            gammaL = gamma_exomol(
                P, T, self.mdb.n_Texp, self.mdb.alpha_ref
            ) + gamma_natural(self.mdb.A)
        dv_lines = self.mdb.nu_lines / R
        ngammaL = gammaL / dv_lines

        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, R)
        Sij = line_strength(
            T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt, Tref_original
        )

        ngammaL_grid = ditgrid_log_interval(
            ngammaL, dit_grid_resolution=self.dit_grid_resolution
        )

        if self.alias == "open":
            xsv = xsvector_open_zeroscan(
                cont_nu,
                index_nu,
                R,
                nsigmaD,
                ngammaL,
                Sij,
                self.nu_grid,
                ngammaL_grid,
                self.nu_grid_extended,
                self.filter_length_oneside,
            )
        elif self.alias == "close":
            xsv = xsvector_zeroscan(
                cont_nu,
                index_nu,
                R,
                pmarray,
                nsigmaD,
                ngammaL,
                Sij,
                self.nu_grid,
                ngammaL_grid,
            )
        return xsv

    def setdgm(self, Tarr_list, Parr, Pself_ref=None):
        """_summary_

        Args:
            Tarr_list (1d or 2d array): tempearture array to be tested such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr (1d array): pressure array in bar
            Pself_ref (1d array, optional): self pressure array in bar. Defaults to None. If None Pself = 0.0.

        Returns:
            _type_: dgm (DIT grid matrix) for gammaL
        """
        from exojax.opacity.modit.modit import exomol, hitran
        from exojax.opacity._common.set_ditgrid import (
            minmax_ditgrid_matrix,
            precompute_modit_ditgrid_matrix,
        )

        cont_nu, index_nu, R, pmarray = self.opainfo
        if len(np.shape(Tarr_list)) == 1:
            Tarr_list = np.array([Tarr_list])
        if Pself_ref is None:
            Pself_ref = np.zeros_like(Parr)

        set_dgm_minmax = []
        for Tarr in Tarr_list:
            if self.mdb.dbtype == "exomol":
                SijM, ngammaLM, nsigmaDl = exomol(
                    self.mdb, Tarr, Parr, R, self.mdb.molmass
                )
            elif self.mdb.dbtype == "hitran":
                SijM, ngammaLM, nsigmaDl = hitran(
                    self.mdb, Tarr, Parr, Pself_ref, R, self.mdb.molmass
                )
            set_dgm_minmax.append(
                minmax_ditgrid_matrix(ngammaLM, self.dit_grid_resolution)
            )
        dgm_ngammaL = precompute_modit_ditgrid_matrix(
            set_dgm_minmax, dit_grid_resolution=self.dit_grid_resolution
        )
        self.dgm_ngammaL = jnp.array(dgm_ngammaL)

    def xsmatrix(self, Tarr, Parr):
        """cross section matrix

        Notes:
            Currently Pself is regarded to be zero for HITEMP/HITRAN

        Args:
            Tarr (): tempearture array in K
            Parr (): pressure array in bar

        Raises:
            ValueError: _description_

        Returns:
            jnp.array : cross section matrix (Nlayer, N_wavenumber)
        """
        from exojax.opacity.modit.modit import (
            exomol,
            hitran,
            xsmatrix_open_zeroscan,
            xsmatrix_zeroscan,
        )

        cont_nu, index_nu, R, pmarray = self.opainfo

        if self.mdb.dbtype == "hitran":
            # qtarr = vmap(self.mdb.qr_interp, (None, 0))(self.mdb.isotope, Tarr)
            SijM, ngammaLM, nsigmaDl = hitran(
                self.mdb, Tarr, Parr, np.zeros_like(Parr), R, self.mdb.molmass
            )
        elif self.mdb.dbtype == "exomol":
            # qtarr = vmap(self.mdb.qr_interp)(Tarr)
            SijM, ngammaLM, nsigmaDl = exomol(self.mdb, Tarr, Parr, R, self.mdb.molmass)
        if self.alias == "open":
            xsm = xsmatrix_open_zeroscan(
                cont_nu,
                index_nu,
                R,
                nsigmaDl,
                ngammaLM,
                SijM,
                self.nu_grid,
                self.dgm_ngammaL,
                self.nu_grid_extended,
                self.filter_length_oneside,
            )

        elif self.alias == "close":
            xsm = xsmatrix_zeroscan(
                cont_nu,
                index_nu,
                R,
                pmarray,
                nsigmaDl,
                ngammaLM,
                SijM,
                self.nu_grid,
                self.dgm_ngammaL,
            )

        return xsm
