"""opacity calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""

__all__ = ['OpaPremodit', 'OpaModit', 'OpaDirect']

from exojax.spec import initspec
from exojax.spec.lbderror import optimal_params
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import nx_from_resolution_eslog
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
from exojax.utils.constants import Patm
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpy as np
import warnings


class OpaCalc():
    """Common Opacity Calculator Class
    """
    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.opainfo = None
        self.method = None  # which opacity calc method is used
        self.ready = False  # ready for opacity computation


class OpaPremodit(OpaCalc):
    """Opacity Calculator Class for PreMODIT

    Attributes:
        opainfo: information set used in PreMODIT

    """
    def __init__(self,
                 mdb,
                 nu_grid,
                 diffmode=2,
                 auto_trange=None,
                 manual_params=None,
                 dit_grid_resolution=0.2):
        """initialization of OpaPremodit

        Note:
            If auto_trange nor manual_params is not given in arguments, 
            use manual_setting()
            or provide self.dE, self.Tref, self.Twt and apply self.apply_params()
            

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            diffmode (int, optional): _description_. Defaults to 2.
            auto_trange (optional): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            manual_params (optional): premodit param set [dE, Tref, Twt]. Defaults to None.
        """
        super().__init__()

        #default setting
        self.method = "premodit"
        self.diffmode = diffmode
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.dit_grid_resolution = dit_grid_resolution
        if auto_trange is not None:
            self.auto_setting(auto_trange[0], auto_trange[1])
        elif manual_params is not None:
            self.manual_setting(manual_params[0], manual_params[1],
                                manual_params[2])
        else:
            print("OpaPremodit: init w/o params setting")
            print("Call self.apply_params() to complete the setting.")

    def auto_setting(self, Tl, Tu):
        print("OpaPremodit: params automatically set.")
        self.dE, self.Tref, self.Twt = optimal_params(Tl, Tu, self.diffmode)
        self.Tmax = Tu
        self.apply_params()

    def manual_setting(self, dE, Tref, Twt):
        """setting PreMODIT parameters by manual

        Args:
            dE (float): E lower grid interval (cm-1)
            Tref (float): reference temperature (K)
            Twt (float): Temperature for weight (K)
        """
        print("OpaPremodit: params manually set.")
        self.Twt = Twt
        self.Tref = Tref
        self.dE = dE
        self.Tmax = np.max([Twt, Tref])
        self.apply_params()

    def set_nu_grid(self, x0, x1, unit, resolution=700000, Nx=None):
        if Nx is None:
            Nx = nx_from_resolution_eslog(x0, x1, resolution)
        if np.mod(Nx, 2) == 1:
            Nx = Nx + 1
        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            x0, x1, Nx, unit=unit, xsmode="premodit")

    def set_gamma_and_n_Texp(self, mdb):
        if mdb.dbtype == "hitran":
            print("gamma_air and n_air are used. gamma_ref = gamma_air/Patm")
            self.gamma_ref = mdb.gamma_air / Patm
            self.n_Texp = mdb.n_air
        elif mdb.dbtype == "exomol":
            self.gamma_ref = mdb.alpha_ref
            self.n_Texp = mdb.n_Texp

    def apply_params(self):
        self.mdb.change_reference_temperature(self.Tref)
        self.dbtype = self.mdb.dbtype
        self.set_gamma_and_n_Texp(self.mdb)
        self.opainfo = initspec.init_premodit(
            self.mdb.nu_lines,
            self.nu_grid,
            self.mdb.elower,
            self.gamma_ref,
            self.n_Texp,
            self.mdb.line_strength_ref,
            self.Twt,
            Tref=self.Tref,
            Tmax=self.Tmax,
            dE=self.dE,
            dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            warning=self.warning)
        self.ready = True

    def xsvector(self, T, P):
        from exojax.spec.premodit import xsvector_zeroth
        from exojax.spec.premodit import xsvector_first
        from exojax.spec.premodit import xsvector_second
        from exojax.spec import normalized_doppler_sigma

        lbd_coeff, multi_index_uniqgrid, elower_grid, \
            ngamma_ref_grid, n_Texp_grid, R, pmarray = self.opainfo
        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, R)

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T)

        if self.diffmode == 0:
            return xsvector_zeroth(T, P, nsigmaD, lbd_coeff, self.Tref, R,
                                   pmarray, self.nu_grid, elower_grid,
                                   multi_index_uniqgrid, ngamma_ref_grid,
                                   n_Texp_grid, qt)
        elif self.diffmode == 1:
            return xsvector_first(T, P, nsigmaD, lbd_coeff, self.Tref,
                                  self.Twt, R, pmarray, self.nu_grid,
                                  elower_grid, multi_index_uniqgrid,
                                  ngamma_ref_grid, n_Texp_grid, qt)
        elif self.diffmode == 2:
            return xsvector_second(T, P, nsigmaD, lbd_coeff, self.Tref,
                                   self.Twt, R, pmarray, self.nu_grid,
                                   elower_grid, multi_index_uniqgrid,
                                   ngamma_ref_grid, n_Texp_grid, qt)

    def xsmatrix(self, Tarr, Parr):
        """cross section matrix

        Args:
            Tarr (): tempearture array in K 
            Parr (): pressure array in bar

        Raises:
            ValueError: _description_

        Returns:
            jnp array: cross section array
        """
        from exojax.spec.premodit import xsmatrix_zeroth
        from exojax.spec.premodit import xsmatrix_first
        from exojax.spec.premodit import xsmatrix_second
        from jax import vmap
        lbd_coeff, multi_index_uniqgrid, elower_grid, \
            ngamma_ref_grid, n_Texp_grid, R, pmarray = self.opainfo

        if self.mdb.dbtype == "hitran":
            qtarr = vmap(self.mdb.qr_interp, (None, 0))(self.mdb.isotope, Tarr)
        elif self.mdb.dbtype == "exomol":
            qtarr = vmap(self.mdb.qr_interp)(Tarr)

        if self.diffmode == 0:
            return xsmatrix_zeroth(Tarr, Parr, self.Tref, R, pmarray,
                                   lbd_coeff, self.nu_grid, ngamma_ref_grid,
                                   n_Texp_grid, multi_index_uniqgrid,
                                   elower_grid, self.mdb.molmass, qtarr)

        elif self.diffmode == 1:
            return xsmatrix_first(Tarr, Parr, self.Tref, self.Twt, R, pmarray,
                                  lbd_coeff, self.nu_grid, ngamma_ref_grid,
                                  n_Texp_grid, multi_index_uniqgrid,
                                  elower_grid, self.mdb.molmass, qtarr)

        elif self.diffmode == 2:
            return xsmatrix_second(Tarr, Parr, self.Tref, self.Twt, R, pmarray,
                                   lbd_coeff, self.nu_grid, ngamma_ref_grid,
                                   n_Texp_grid, multi_index_uniqgrid,
                                   elower_grid, self.mdb.molmass, qtarr)

        else:
            raise ValueError("diffmode should be 0, 1, 2.")


class OpaModit(OpaCalc):
    """Opacity Calculator Class for MODIT

    Attributes:
        opainfo: information set used in MODIT: cont_nu, index_nu, R, pmarray

    """
    def __init__(self,
                 mdb,
                 nu_grid,
                 Tarr_list=None,
                 Parr=None,
                 Pself_ref=None,
                 dit_grid_resolution=0.2):
        """initialization of OpaModit

        Note:
            Tarr_list and Parr are used to compute xsmatrix. No need for xsvector

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            Tarr_list (1d or 2d array, optional): tempearture array to be tested such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr (1d array, optional): pressure array in bar
            Pself_ref (1d array, optional): self pressure array in bar. Defaults to None. If None Pself = 0.0.
            dit_grid_resolution (float, optional): dit grid resolution. Defaults to 0.2.

        Raises:
            ValueError: _description_
        """
        super().__init__()

        #default setting
        self.method = "modit"
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.dit_grid_resolution = dit_grid_resolution
        if not self.mdb.gpu_transfer:
            raise ValueError("For MODIT, gpu_transfer should be True in mdb.")
        self.apply_params()
        if Tarr_list is not None and Parr is not None:
            self.setdgm(Tarr_list, Parr, Pself_ref=Pself_ref)
        else:
            warnings.warn("Tarr_list/Parr are needed for xsmatrix.",
                          UserWarning)

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
        from exojax.spec import normalized_doppler_sigma, gamma_natural
        from exojax.spec.hitran import line_strength
        from exojax.spec.exomol import gamma_exomol
        from exojax.spec.hitran import gamma_hitran
        from exojax.spec.set_ditgrid import ditgrid_log_interval
        from exojax.spec.modit_scanfft import xsvector_scanfft
        from exojax.spec import normalized_doppler_sigma

        cont_nu, index_nu, R, pmarray = self.opainfo

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air,
                self.mdb.gamma_self) + gamma_natural(self.mdb.A)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T)
            gammaL = gamma_exomol(P, T, self.mdb.n_Texp,
                                  self.mdb.alpha_ref) + gamma_natural(
                                      self.mdb.A)

        dv_lines = self.mdb.nu_lines / R
        ngammaL = gammaL / dv_lines
        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, R)
        Sij = line_strength(T, self.mdb.logsij0, self.mdb.nu_lines,
                            self.mdb.elower, qt)

        ngammaL_grid = ditgrid_log_interval(
            ngammaL, dit_grid_resolution=self.dit_grid_resolution)
        return xsvector_scanfft(cont_nu, index_nu, R, pmarray, nsigmaD,
                                ngammaL, Sij, self.nu_grid, ngammaL_grid)

    def setdgm(self, Tarr_list, Parr, Pself_ref=None):
        """_summary_

        Args:
            Tarr_list (1d or 2d array): tempearture array to be tested such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr (1d array): pressure array in bar
            Pself_ref (1d array, optional): self pressure array in bar. Defaults to None. If None Pself = 0.0.

        Returns:
            _type_: dgm (DIT grid matrix) for gammaL
        """
        from exojax.spec.set_ditgrid import minmax_ditgrid_matrix
        from exojax.spec.set_ditgrid import precompute_modit_ditgrid_matrix
        from exojax.spec.modit import hitran
        from exojax.spec.modit import exomol

        cont_nu, index_nu, R, pmarray = self.opainfo
        if len(np.shape(Tarr_list)) == 1:
            Tarr_list = np.array([Tarr_list])
        if Pself_ref is None:
            Pself_ref = np.zeros_like(Parr)

        set_dgm_minmax = []
        for Tarr in Tarr_list:
            if self.mdb.dbtype == "exomol":
                SijM, ngammaLM, nsigmaDl = exomol(self.mdb, Tarr, Parr, R,
                                                  self.mdb.molmass)
            elif self.mdb.dbtype == "hitran":
                SijM, ngammaLM, nsigmaDl = hitran(self.mdb, Tarr, Parr,
                                                  Pself_ref, R,
                                                  self.mdb.molmass)
            set_dgm_minmax.append(
                minmax_ditgrid_matrix(ngammaLM, self.dit_grid_resolution))
        dgm_ngammaL = precompute_modit_ditgrid_matrix(
            set_dgm_minmax, dit_grid_resolution=self.dit_grid_resolution)
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
            jnp array: cross section array
        """
        from exojax.spec.modit_scanfft import xsmatrix_scanfft
        from exojax.spec.modit import exomol
        from exojax.spec.modit import hitran
        cont_nu, index_nu, R, pmarray = self.opainfo

        if self.mdb.dbtype == "hitran":
            #qtarr = vmap(self.mdb.qr_interp, (None, 0))(self.mdb.isotope, Tarr)
            SijM, ngammaLM, nsigmaDl = hitran(self.mdb, Tarr, Parr,
                                              np.zeros_like(Parr), R,
                                              self.mdb.molmass)
        elif self.mdb.dbtype == "exomol":
            #qtarr = vmap(self.mdb.qr_interp)(Tarr)
            SijM, ngammaLM, nsigmaDl = exomol(self.mdb, Tarr, Parr, R,
                                              self.mdb.molmass)

        return xsmatrix_scanfft(cont_nu, index_nu, R, pmarray, nsigmaDl,
                                ngammaLM, SijM, self.nu_grid, self.dgm_ngammaL)


class OpaDirect(OpaCalc):
    def __init__(self, mdb, nu_grid):
        """initialization of OpaDirect (LPF)

            

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
        """
        super().__init__()

        #default setting
        self.method = "lpf"
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.mdb = mdb
        self.apply_params()

    def apply_params(self):
        self.dbtype = self.mdb.dbtype
        self.opainfo = initspec.init_lpf(self.mdb.nu_lines, self.nu_grid)
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
        from exojax.spec import gamma_natural
        from exojax.spec import doppler_sigma
        from exojax.spec.exomol import gamma_exomol
        from exojax.spec.hitran import gamma_hitran
        from exojax.spec.hitran import line_strength
        from exojax.spec.lpf import xsvector as xsvector_lpf

        numatrix = self.opainfo

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air,
                self.mdb.gamma_self) + gamma_natural(self.mdb.A)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T)
            gammaL = gamma_exomol(P, T, self.mdb.n_Texp,
                                  self.mdb.alpha_ref) + gamma_natural(
                                      self.mdb.A)
        sigmaD = doppler_sigma(self.mdb.nu_lines, T, self.mdb.molmass)
        Sij = line_strength(T, self.mdb.logsij0, self.mdb.nu_lines,
                            self.mdb.elower, qt)
        return xsvector_lpf(numatrix, sigmaD, gammaL, Sij)

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
            jnp array: cross section array
        """
        from exojax.spec import gamma_natural
        from exojax.spec import doppler_sigma
        from exojax.spec.exomol import gamma_exomol
        from exojax.spec.hitran import gamma_hitran
        from exojax.spec.hitran import line_strength
        from exojax.spec.lpf import xsmatrix as xsmatrix_lpf

        numatrix = self.opainfo
        vmaplinestrengh = jit(vmap(line_strength, (0, None, None, None, 0)))
        if self.mdb.dbtype == "hitran":
            vmapqt = vmap(self.mdb.qr_interp, (None, 0))
            qt = vmapqt(self.mdb.isotope, Tarr)
            vmaphitran = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))
            gammaLM = vmaphitran(Parr, Tarr, np.zeros_like(Parr),
                                 self.mdb.n_air, self.mdb.gamma_air,
                                 self.mdb.gamma_self) + gamma_natural(
                                     self.mdb.A)
            SijM = vmaplinestrengh(Tarr, self.mdb.logsij0, self.mdb.nu_lines,
                                   self.mdb.elower, qt)
        elif self.mdb.dbtype == "exomol":
            vmapqt = vmap(self.mdb.qr_interp)
            qt = vmapqt(Tarr)
            vmapexomol = jit(vmap(gamma_exomol, (0, 0, None, None)))
            gammaLMP = vmapexomol(Parr, Tarr, self.mdb.n_Texp,
                                  self.mdb.alpha_ref)
            gammaLMN = gamma_natural(self.mdb.A)
            gammaLM = gammaLMP + gammaLMN[None, :]
            SijM = vmaplinestrengh(Tarr, self.mdb.logsij0, self.mdb.nu_lines,
                                   self.mdb.elower, qt)
        sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(self.mdb.nu_lines,
                                                            Tarr, self.mdb.molmass)
        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)