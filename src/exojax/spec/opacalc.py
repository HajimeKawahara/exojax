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
from exojax.utils.constants import Tref_original
from exojax.utils.jaxstatus import check_jax64bit
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpy as np
import warnings


class OpaCalc():
    """Common Opacity Calculator Class
    """
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
                 diffmode=0,
                 broadening_resolution={
                     "mode": "manual",
                     "value": 0.2
                 },
                 auto_trange=None,
                 manual_params=None,
                 dit_grid_resolution=None,
                 allow_32bit=False,
                 wavelength_order="descending"):
        """initialization of OpaPremodit

        Note:
            If auto_trange nor manual_params is not given in arguments, 
            use manual_setting()
            or provide self.dE, self.Tref, self.Twt and apply self.apply_params()
            
        Note:
            The option of "broadening_parameter_resolution" controls the resolution of broadening parameters.
            When you wanna use the manual resolution, set broadening_parameter_resolution = {mode: "manual", value: 0.2}.
            When you wanna use the min and max values of broadening parameters in database, set broadening_parameter_resolution = {mode: "minmax", value: None}.
            When you wanna give single broadening parameters: set broadening_parameter_resolution = {mode: "single", value: None} the median values of gamma_ref, n_Texp are used
            or set broadening_parameter_resolution = {mode: "single", value: [gamma_ref, n_Texp]} values are at 296K, for the fixed parameter set.  
            The use of device memory: "manual" >= "minmax" > "single". In general, small value (such as 0.2) requires large device memory. 
            We recommend to check the difference of the final specrum between "manual", "minmax", and "single" when you had a device memory problem.
            
        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            diffmode (int, optional): _description_. Defaults to 0.
            broadening_resolution (dict, optional): definition of the broadening parameter resolution. Default to {"mode": "manual", value: 0.2}. See Note. 
            auto_trange (optional): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            manual_params (optional): premodit parameter set [dE, Tref, Twt]. Defaults to None.
            dit_grid_resolution (float, optional): force to set broadening_parameter_resolution={mode:manual, value: dit_grid_resolution}), ignores broadening_parameter_resolution.
            allow_32bit (bool, optional): If True, allow 32bit mode of JAX. Defaults to False.
            wavlength order: wavelength order: "ascending" or "descending"
        """
        super().__init__()
        check_jax64bit(allow_32bit)

        #default setting
        self.method = "premodit"
        self.diffmode = diffmode
        self.warning = True
        self.nu_grid = nu_grid
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(self.nu_grid,
                          wavelength_order=self.wavelength_order,
                          unit="AA")
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.ngrid_broadpar = None

        #broadening parameter setting
        self.determine_broadening_parameter_resolution(broadening_resolution,
                                                       dit_grid_resolution)
        self.broadening_parameters_setting()

        if auto_trange is not None:
            self.auto_setting(auto_trange[0], auto_trange[1])
        elif manual_params is not None:
            self.manual_setting(manual_params[0], manual_params[1],
                                manual_params[2])
        else:
            print("OpaPremodit: initialization without parameters setting")
            print("Call self.apply_params() to complete the setting.")

    def auto_setting(self, Tl, Tu):
        print("OpaPremodit: params automatically set.")
        self.dE, self.Tref, self.Twt = optimal_params(Tl, Tu, self.diffmode)
        self.Tmax = Tu
        self.Tmin = Tl
        self.apply_params()

    def manual_setting(self, dE, Tref, Twt, Tmax=None, Tmin=None):
        """setting PreMODIT parameters by manual

        Args:
            dE (float): E lower grid interval (cm-1)
            Tref (float): reference temperature (K)
            Twt (float): Temperature for weight (K)
            Tmax (float/None): max temperature (K) for braodening grid
            Tmin (float/None): min temperature (K) for braodening grid
        """
        print("OpaPremodit: params manually set.")
        self.Twt = Twt
        self.Tref = Tref
        self.dE = dE
        if Tmax is None:
            Tmax = np.max([Twt, Tref])
        if Tmin is None:
            Tmin = np.min([Twt, Tref])

        self.Tmax = Tmax
        self.Tmin = Tmin
        self.apply_params()

    def set_nu_grid(self, x0, x1, unit, resolution=700000, Nx=None):
        if Nx is None:
            Nx = nx_from_resolution_eslog(x0, x1, resolution)
        if np.mod(Nx, 2) == 1:
            Nx = Nx + 1
        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            x0, x1, Nx, unit=unit, xsmode="premodit")

    def set_Tref_broadening_to_midpoint(self):
        """Set self.Tref_broadening using log midpoint of Tmax and Tmin
        """
        from exojax.spec.premodit import reference_temperature_broadening_at_midpoint
        self.Tref_broadening = reference_temperature_broadening_at_midpoint(
            self.Tmin, self.Tmax)
        print("OpaPremodit: Tref_broadening is set to ", self.Tref_broadening,
              "K")

    def determine_broadening_parameter_resolution(
            self, broadening_parameter_resolution, dit_grid_resolution):
        if dit_grid_resolution is not None:
            warnings.warn(
                "dit_grid_resolution is not None. Ignoring broadening_parameter_resolution.",
                UserWarning)
            self.broadening_parameter_resolution = {
                "mode": "manual",
                "value": dit_grid_resolution
            }
        else:
            self.broadening_parameter_resolution = broadening_parameter_resolution

    def broadening_parameters_setting(self):
        mode = self.broadening_parameter_resolution["mode"]
        val = self.broadening_parameter_resolution["value"]
        if mode == "manual":
            self.dit_grid_resolution = val
            self.single_broadening = False
            self.single_broadening_parameters = None
        elif mode == "single":
            self.dit_grid_resolution = None
            self.single_broadening = True
            if val is None:
                val = [None, None]
            self.single_broadening_parameters = val
        elif mode == "minmax":
            self.dit_grid_resolution = np.inf
            self.single_broadening = False
            self.single_broadening_parameters = None
        else:
            raise ValueError(
                "Unknown mode in broadening_parameter_resolution e.g. manual/single/minmax."
            )

    def compute_gamma_ref_and_n_Texp(self, mdb):
        """convert gamma_ref to the regular formalization and noramlize it for Tref_braodening

        Notes:
            gamma (T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)**n 
            * (T/Tref_broadening)**-n * (P/1bar) 

        Args:
            mdb (_type_): mdb instance

        """
        if mdb.dbtype == "hitran":
            print(
                "OpaPremodit: gamma_air and n_air are used. gamma_ref = gamma_air/Patm"
            )
            self.n_Texp = mdb.n_air
            reference_factor = (Tref_original /
                                self.Tref_broadening)**(self.n_Texp)
            self.gamma_ref = mdb.gamma_air * reference_factor / Patm
        elif mdb.dbtype == "exomol":
            self.n_Texp = mdb.n_Texp
            reference_factor = (Tref_original /
                                self.Tref_broadening)**(self.n_Texp)
            self.gamma_ref = mdb.alpha_ref * reference_factor

    def apply_params(self):
        self.mdb.change_reference_temperature(self.Tref)
        self.dbtype = self.mdb.dbtype

        #broadening
        if self.single_broadening:
            print("OpaPremodit: a single broadening parameter set is used.")
            self.Tref_broadening = Tref_original
        else:
            self.set_Tref_broadening_to_midpoint()

        self.compute_gamma_ref_and_n_Texp(self.mdb)

        self.opainfo = initspec.init_premodit(
            self.mdb.nu_lines,
            self.nu_grid,
            self.mdb.elower,
            self.gamma_ref,
            self.n_Texp,
            self.mdb.line_strength_ref,
            self.Twt,
            Tref=self.Tref,
            Tref_broadening=self.Tref_broadening,
            Tmax=self.Tmax,
            Tmin=self.Tmin,
            dE=self.dE,
            dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            single_broadening=self.single_broadening,
            single_broadening_parameters=self.single_broadening_parameters,
            warning=self.warning)
        self.ready = True

        lbd_coeff, multi_index_uniqgrid, elower_grid, \
            ngamma_ref_grid, n_Texp_grid, R, pmarray = self.opainfo
        self.ngrid_broadpar = len(multi_index_uniqgrid)
        self.ngrid_elower = len(elower_grid)

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
                                   n_Texp_grid, qt, self.Tref_broadening)
        elif self.diffmode == 1:
            return xsvector_first(T, P, nsigmaD, lbd_coeff, self.Tref,
                                  self.Twt, R, pmarray, self.nu_grid,
                                  elower_grid, multi_index_uniqgrid,
                                  ngamma_ref_grid, n_Texp_grid, qt,
                                  self.Tref_broadening)
        elif self.diffmode == 2:
            return xsvector_second(T, P, nsigmaD, lbd_coeff, self.Tref,
                                   self.Twt, R, pmarray, self.nu_grid,
                                   elower_grid, multi_index_uniqgrid,
                                   ngamma_ref_grid, n_Texp_grid, qt,
                                   self.Tref_broadening)

    def xsmatrix(self, Tarr, Parr):
        """cross section matrix

        Args:
            Tarr (): tempearture array in K 
            Parr (): pressure array in bar

        Raises:
            ValueError: _description_

        Returns:
            jnp.array : cross section matrix (Nlayer, N_wavenumber)
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
                                   elower_grid, self.mdb.molmass, qtarr,
                                   self.Tref_broadening)

        elif self.diffmode == 1:
            return xsmatrix_first(Tarr, Parr, self.Tref, self.Twt, R, pmarray,
                                  lbd_coeff, self.nu_grid, ngamma_ref_grid,
                                  n_Texp_grid, multi_index_uniqgrid,
                                  elower_grid, self.mdb.molmass, qtarr,
                                  self.Tref_broadening)

        elif self.diffmode == 2:
            return xsmatrix_second(Tarr, Parr, self.Tref, self.Twt, R, pmarray,
                                   lbd_coeff, self.nu_grid, ngamma_ref_grid,
                                   n_Texp_grid, multi_index_uniqgrid,
                                   elower_grid, self.mdb.molmass, qtarr,
                                   self.Tref_broadening)

        else:
            raise ValueError("diffmode should be 0, 1, 2.")

    def plot_broadening_parameters(self,
                                   figname="broadpar_grid.png",
                                   crit=300000):
        """plot broadening parameters and grids

        Args:
            figname (str, optional): output image file. Defaults to "broadpar_grid.png".
            crit (int, optional): sampling criterion. Defaults to 300000. when the number of lines is huge and if it exceeded ~ crit, we sample the lines to reduce the computation.
        """
        from exojax.plot.opaplot import plot_broadening_parameters_grids
        _, _, _, ngamma_ref_grid, n_Texp_grid, _, _ = self.opainfo
        gamma_ref_in = self.gamma_ref
        n_Texp_in = self.n_Texp
        plot_broadening_parameters_grids(ngamma_ref_grid, n_Texp_grid,
                                         self.nu_grid, self.resolution,
                                         gamma_ref_in, n_Texp_in, crit,
                                         figname)


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
                 dit_grid_resolution=0.2,
                 allow_32bit=False,
                 wavelength_order="descending"):
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
            wavlength order: wavelength order: "ascending" or "descending"
            
        Raises:
            ValueError: _description_
        """
        super().__init__()
        check_jax64bit(allow_32bit)

        #default setting
        self.method = "modit"
        self.warning = True
        self.nu_grid = nu_grid
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(self.nu_grid,
                          wavelength_order=self.wavelength_order,
                          unit="AA")
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
            jnp.array : cross section matrix (Nlayer, N_wavenumber)
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
    def __init__(self, mdb, nu_grid, wavelength_order="descending"):
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
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(self.nu_grid,
                          wavelength_order=self.wavelength_order,
                          unit="AA")
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
            jnp.array : cross section matrix (Nlayer, N_wavenumber)
        """
        from exojax.spec import gamma_natural
        from exojax.spec import doppler_sigma
        from exojax.spec.exomol import gamma_exomol
        from exojax.spec.hitran import gamma_hitran
        from exojax.spec.hitran import line_strength
        from exojax.spec.atomll import gamma_vald3
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
            sigmaDM = jit(vmap(doppler_sigma,
                            (None, 0, None)))(self.mdb.nu_lines, Tarr,
                                                self.mdb.molmass)
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
            sigmaDM = jit(vmap(doppler_sigma,
                            (None, 0, None)))(self.mdb.nu_lines, Tarr,
                                                self.mdb.molmass)
        elif (self.mdb.dbtype == "kurucz") or (self.mdb.dbtype == "vald"):
            qt_284=vmap(self.mdb.QT_interp_284)(Tarr)
            qt_K = jnp.zeros([len(self.mdb.QTmask), len(Tarr)])
            for i, mask in enumerate(self.mdb.QTmask):
                qt_K = qt_K.at[i].set(qt_284[:,mask]) #e.g., qt_284[:,76] #Fe I
            qt_K = jnp.array(qt_K)     
            vmapvald3 = jit(vmap(gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))
            PH,PHe,PHH = Parr*self.mdb.vmrH, Parr*self.mdb.vmrHe, Parr*self.mdb.vmrHH 
            gammaLM = vmapvald3(Tarr, PH, PHH, PHe, self.mdb.ielem, self.mdb.iion, \
                                self.mdb.dev_nu_lines, self.mdb.elower, self.mdb.eupper, \
                                self.mdb.atomicmass, self.mdb.ionE, \
                                self.mdb.gamRad, self.mdb.gamSta, self.mdb.vdWdamp, 1.0)
            SijM = vmaplinestrengh(Tarr, self.mdb.logsij0, self.mdb.nu_lines, \
                                    self.mdb.elower, qt_K.T)  
            sigmaDM = jit(vmap(doppler_sigma,(None,0,None)))\
                (self.mdb.nu_lines, Tarr, self.mdb.atomicmass)

        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)