"""opacity calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""

__all__ = ["OpaPremodit", "OpaModit", "OpaDirect"]

from exojax.spec import initspec
from exojax.spec.lbderror import optimal_params
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import nx_even_from_resolution_eslog
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
from exojax.utils.constants import Patm
from exojax.utils.constants import Tref_original
from exojax.utils.jaxstatus import check_jax64bit
from exojax.utils.checkarray import is_outside_range
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
from exojax.signal.ola import overlap_and_add
from exojax.signal.ola import ola_output_length
from exojax.signal.ola import overlap_and_add_matrix
import jax.numpy as jnp
from jax.lax import scan
from jax.lax import dynamic_slice
from jax import jit
from jax import vmap

import numpy as np
import warnings


class OpaCalc:
    """Common Opacity Calculator Class

    Attributes:
        opainfo: information set used in each opacity method
        method (str,None): opacity calculation method, i.e. "premodit", "modit", "lpf"
        ready (bool): ready for opacity computation
        alias (bool): mode of the aliasing part for the convolution (MODIT/PreMODIT).
            False = the closed mode, left and right alising sides are overlapped and won't be used.
            True = the open mode, left and right aliasing sides are not overlapped and the alias part will be used in OLA.
        nu_grid_extended (jnp.array): extended wavenumber grid for the open mode
        filter_length_oneside (int): oneside number of points to be added to the left and right of the nu_grid based on the cutwing ratio
        filter_length (int): total number of points to be added to the left and right of the nu_grid based on the cutwing ratio
        cutwing (float): wingcut for the convolution used in open cross section. Defaults to 1.0. For alias="close", always 1.0 is used by definition.
        wing_cut_width (list): min and max wing cut width in cm-1


    """

    def __init__(self, nu_grid):
        self.nu_grid = nu_grid
        self.opainfo = None
        self.method = None  # which opacity calc method is used
        self.ready = False  # ready for opacity computation
        self.alias = "close"  # close or open
        self.nstitch = 1

        # open xsvector/xsmatrix
        self.cutwing = 1.0
        self.nu_grid_extended = None
        self.filter_length_oneside = 0

    def set_aliasing(self):
        """set the aliasing

        Raises:
            ValueError: alias should be 'close' or 'open'
        """
        from exojax.utils.grids import extended_wavenumber_grid

        self.set_filter_length_oneside_from_cutwing()

        if self.nstitch > 1:
            print("cross section is calculated in the stitching mode.")
            self.nu_grid_array = np.array(np.array_split(self.nu_grid, self.nstitch))
            self.nu_grid_extended_array = []
            for i in range(self.nstitch):
                self.nu_grid_extended_array.append(
                    extended_wavenumber_grid(
                        self.nu_grid_array[i, :],
                        self.filter_length_oneside,
                        self.filter_length_oneside,
                    )
                )
            self.nu_grid_extended_array = np.array(self.nu_grid_extended_array)
            self.wing_cut_width = [
                self.nu_grid[0] - self.nu_grid_extended_array[0, 0],
                self.nu_grid_extended_array[-1, -1] - self.nu_grid[-1],
            ]
        elif self.alias == "close":
            print(
                "cross section (xsvector/xsmatrix) is calculated in the closed mode. The aliasing part cannnot be used."
            )
            resolution = resolution_eslog(self.nu_grid)
            lnx0 = np.log10(self.nu_grid[0]) - len(self.nu_grid) / resolution / np.log(
                10
            )
            lnx1 = np.log10(self.nu_grid[-1]) + len(self.nu_grid) / resolution / np.log(
                10
            )
            self.wing_cut_width = [
                self.nu_grid[0] - 10**lnx0,
                10**lnx1 - self.nu_grid[-1],
            ]
        elif self.alias == "open":
            print(
                "cross section (xsvector/xsmatrix) is calculated in the open mode. The aliasing part can be used."
            )

            self.nu_grid_extended = extended_wavenumber_grid(
                self.nu_grid, self.filter_length_oneside, self.filter_length_oneside
            )
            self.wing_cut_width = [
                self.nu_grid[0] - self.nu_grid_extended[0],
                self.nu_grid_extended[-1] - self.nu_grid[-1],
            ]

        else:
            raise ValueError(
                "nstitch > 1 or when nstitch =1 then alias should be 'close' or 'open'."
            )

        print("wing cut width = ", self.wing_cut_width, "cm-1")

    def set_filter_length_oneside_from_cutwing(self):
        """sets the number of points to be added to the left and right (filter_lenth_oneside) of the nu_grid based on the cutwing ratio"""
        self.div_length = len(self.nu_grid) // self.nstitch
        self.filter_length_oneside = int(len(self.nu_grid) * self.cutwing)
        self.filter_length = 2 * self.filter_length_oneside + 1
        self.output_length = ola_output_length(
            self.nstitch, self.div_length, self.filter_length
        )

    def check_nu_grid_reducible(self):
        """check if nu_grid is reducible by ndiv

        Raises:
            ValueError: if nu_grid is not reducible by ndiv
        """
        if len(self.nu_grid) % self.nstitch != 0:
            msg = (
                "nu_grid_all length = "
                + str(len(self.nu_grid))
                + " cannot be divided by stitch="
                + str(self.nstitch)
            )
            raise ValueError(msg)


class OpaPremodit(OpaCalc):
    """Opacity Calculator Class for PreMODIT

    Attributes:
        opainfo: information set used in PreMODIT

    """

    def __init__(
        self,
        mdb,
        nu_grid,
        diffmode=0,
        broadening_resolution={"mode": "manual", "value": 0.2},
        auto_trange=None,
        manual_params=None,
        dit_grid_resolution=None,
        allow_32bit=False,
        nstitch=1,
        cutwing=1.0,
        wavelength_order="descending",
        version_auto_trange=2,
    ):
        """initialization of OpaPremodit

        Note:
            If auto_trange nor manual_params is not given in arguments,
            use manual_setting()
            or provide self.dE, self.Twt and apply self.apply_params()

        Note:
            The option of "broadening_resolution" controls the resolution of broadening parameters.
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
            dit_grid_resolution (float, optional): force to set broadening_resolution={mode:manual, value: dit_grid_resolution}), ignores broadening_resolution.
            allow_32bit (bool, optional): If True, allow 32bit mode of JAX. Defaults to False.
            nstitch (int, optional): number of stitching. Defaults to 1.
            cutwing (float, optional): wingcut for the convolution used when nstitch > 1. Defaults to 1.0.
            wavlength order: wavelength order: "ascending" or "descending"
            version_auto_trange: version of the default elower grid trange (degt) file, Default to 2 since Jan 2024.
        """
        super().__init__(nu_grid)
        check_jax64bit(allow_32bit)

        # default setting
        self.method = "premodit"
        self.diffmode = diffmode
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.ngrid_broadpar = None
        self.version_auto_trange = version_auto_trange
        # check if the mdb lines are in nu_grid
        if is_outside_range(self.mdb.nu_lines, self.nu_grid[0], self.nu_grid[-1]):
            raise ValueError("None of the lines in mdb are within nu_grid.")

        self.determine_broadening_parameter_resolution(
            broadening_resolution, dit_grid_resolution
        )
        self.broadening_parameters_setting()

        if auto_trange is not None:
            self.auto_setting(auto_trange[0], auto_trange[1])
        elif manual_params is not None:
            self.manual_setting(manual_params[0], manual_params[1], manual_params[2])
        else:
            print("OpaPremodit: initialization without parameters setting")
            print("Call self.apply_params() to complete the setting.")

        self.nstitch = nstitch
        self.cutwing = cutwing

        if self.nstitch > 1:
            print("OpaPremodit: Stitching mode is used: nstitch =", self.nstitch)
            self.check_nu_grid_reducible()
            self.alias = "open"
        else:
            self.alias = "close"
        self.set_aliasing()

        self._sets_capable_opacalculators()
        if nstitch > 1:
            self.reshape_lbd_coeff()

    def __eq__(self, other):
        """eq method for OpaPremodit, definied by comparing all the attributes and important status

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, OpaPremodit):
            return False

        eq_attributes = (
            (self.mdb == other.mdb)
            and (self.diffmode == other.diffmode)
            and (self.ngrid_broadpar == other.ngrid_broadpar)
            and (self.wavelength_order == other.wavelength_order)
            and (self.version_auto_trange == other.version_auto_trange)
            and all(self.nu_grid == other.nu_grid)
        )
        eq_attributes = self._if_exist_check_eq(other, "dE", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tref", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Twt", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tmax", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tmin", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tref_broadening", eq_attributes)

        return eq_attributes

    def _if_exist_check_eq(self, other, attribute, eq_attributes):
        if hasattr(self, attribute) and hasattr(other, attribute):
            return eq_attributes and getattr(self, attribute) == getattr(
                other, attribute
            )
        elif not hasattr(self, attribute) and not hasattr(other, attribute):
            return eq_attributes
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def auto_setting(self, Tl, Tu):
        print("OpaPremodit: params automatically set.")
        self.dE, self.Tref, self.Twt = optimal_params(
            Tl, Tu, self.diffmode, self.version_auto_trange
        )
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
            Nx = nx_even_from_resolution_eslog(x0, x1, resolution)
        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            x0, x1, Nx, unit=unit, xsmode="premodit"
        )

    def set_Tref_broadening_to_midpoint(self):
        """Set self.Tref_broadening using log midpoint of Tmax and Tmin"""
        from exojax.spec.premodit import reference_temperature_broadening_at_midpoint

        self.Tref_broadening = reference_temperature_broadening_at_midpoint(
            self.Tmin, self.Tmax
        )
        print("OpaPremodit: Tref_broadening is set to ", self.Tref_broadening, "K")

    def determine_broadening_parameter_resolution(
        self, broadening_parameter_resolution, dit_grid_resolution
    ):
        if dit_grid_resolution is not None:
            warnings.warn(
                "dit_grid_resolution is not None. Ignoring broadening_parameter_resolution.",
                UserWarning,
            )
            self.broadening_parameter_resolution = {
                "mode": "manual",
                "value": dit_grid_resolution,
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

    def compute_gamma_ref_and_n_Texp(self):
        """convert gamma_ref to the regular formalization and noramlize it for Tref_braodening

        Notes:
            gamma (T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)**n
            * (T/Tref_broadening)**-n * (P/1bar)

        Args:
            mdb (_type_): mdb instance

        """
        if self.mdb.dbtype == "hitran":
            print(
                "OpaPremodit: gamma_air and n_air are used. gamma_ref = gamma_air/Patm"
            )
            self.n_Texp = self.mdb.n_air
            reference_factor = (Tref_original / self.Tref_broadening) ** (self.n_Texp)
            self.gamma_ref = self.mdb.gamma_air * reference_factor / Patm
        elif self.mdb.dbtype == "exomol":
            self.n_Texp = self.mdb.n_Texp
            reference_factor = (Tref_original / self.Tref_broadening) ** (self.n_Texp)
            self.gamma_ref = self.mdb.alpha_ref * reference_factor

    def apply_params(self):
        """apply the parameters to the class
        define self.lbd_coeff and self.opainfo
        """
        # self.mdb.change_reference_temperature(self.Tref)
        self.dbtype = self.mdb.dbtype

        # sets the broadening reference temperature
        if self.single_broadening:
            print("OpaPremodit: a single broadening parameter set is used.")
            self.Tref_broadening = Tref_original
        else:
            self.set_Tref_broadening_to_midpoint()

        # self.gamma_ref, self.n_Texp are defined with the reference temperature of Tref_broadening
        self.compute_gamma_ref_and_n_Texp()

        # comment-1: gamma_ref at Tref_broadening (is not necessary for Tref_original)
        # comment-2: line strength at Tref (is not necessary for Tref_original)
        (
            self.lbd_coeff,
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        ) = initspec.init_premodit(
            self.mdb.nu_lines,
            self.nu_grid,
            self.mdb.elower,
            self.gamma_ref,  # comment-1
            self.n_Texp,
            self.mdb.line_strength(self.Tref),  # comment-2
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
            warning=self.warning,
        )
        self.opainfo = (
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        )
        self.ready = True

        self.ngrid_broadpar = len(multi_index_uniqgrid)
        self.ngrid_elower = len(elower_grid)

    def _sets_capable_opacalculators(self):
        """sets capable opacalculators"""
        # opa calculators for PreMODIT
        from exojax.spec.premodit import xsvector_zeroth
        from exojax.spec.premodit import xsvector_first
        from exojax.spec.premodit import xsvector_second
        from exojax.spec.premodit import xsmatrix_zeroth
        from exojax.spec.premodit import xsmatrix_first
        from exojax.spec.premodit import xsmatrix_second
        from exojax.spec.premodit import xsvector_nu_open_zeroth
        from exojax.spec.premodit import xsvector_nu_open_first
        from exojax.spec.premodit import xsvector_nu_open_second
        from exojax.spec.premodit import xsmatrix_nu_open_zeroth
        from exojax.spec.premodit import xsmatrix_nu_open_first
        from exojax.spec.premodit import xsmatrix_nu_open_second

        self.xsvector_close = {
            0: xsvector_zeroth,
            1: xsvector_first,
            2: xsvector_second,
        }
        self.xsmatrix_close = {
            0: xsmatrix_zeroth,
            1: xsmatrix_first,
            2: xsmatrix_second,
        }
        self.xsvector_stitch = {
            0: xsvector_nu_open_zeroth,
            1: xsvector_nu_open_first,
            2: xsvector_nu_open_second,
        }
        self.xsmatrix_stitch = {
            0: xsmatrix_nu_open_zeroth,
            1: xsmatrix_nu_open_first,
            2: xsmatrix_nu_open_second,
        }

    def reshape_lbd_coeff(self):
        """reshape lbd_coeff for stitching mode
        this method deletes self.lbd_coeff and creates self.lbd_coeff_reshaped
        self.lbd_coeff_reshaped has a dimension of (self.nstitch, diffmode+1, self.div_length, N_broadening, len(elower_grid))
        """

        shape_lbd = self.lbd_coeff.shape
        lbd_coeff_reshaped = np.zeros(
            (
                self.nstitch,
                shape_lbd[0],
                self.div_length,
                shape_lbd[2],  # N_broadening
                shape_lbd[3],  # N_Elower
            )
        )
        for i in range(self.nstitch):
            lbd_coeff_reshaped[i, ...] = self.lbd_coeff[
                :, i * self.div_length : (i + 1) * self.div_length, ...
            ]
        self.lbd_coeff_reshaped = np.array(lbd_coeff_reshaped)
        del self.lbd_coeff

    def xsvector(self, T, P):
        from exojax.spec import normalized_doppler_sigma

        (
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        ) = self.opainfo
        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, R)

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, self.Tref)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T, self.Tref)

        if self.nstitch > 1:

            def floop(icarry, lbd_coeff):
                nu_grid_each = dynamic_slice(
                    self.nu_grid, (icarry * self.div_length,), (self.div_length,)
                )
                xsv_nu = self.xsvector_stitch[self.diffmode](
                    T,
                    P,
                    nsigmaD,
                    lbd_coeff,
                    self.Tref,
                    R,
                    nu_grid_each,
                    elower_grid,
                    multi_index_uniqgrid,
                    ngamma_ref_grid,
                    n_Texp_grid,
                    qt,
                    self.Tref_broadening,
                    self.filter_length_oneside,
                    self.Twt,
                )

                return icarry + 1, xsv_nu

            _, xsv_matrix = scan(floop, 0, self.lbd_coeff_reshaped)

            xsv_matrix = xsv_matrix / self.nu_grid_extended_array
            xsv_ola_stitch = overlap_and_add(
                xsv_matrix, self.output_length, self.div_length
            )
            xsv = xsv_ola_stitch[
                self.filter_length_oneside : -self.filter_length_oneside
            ]
        
        elif self.nstitch == 1:
            xsvector_func = self.xsvector_close[self.diffmode]
            xsv = xsvector_func(
                T,
                P,
                nsigmaD,
                self.lbd_coeff,
                self.Tref,
                R,
                pmarray,
                self.nu_grid,
                elower_grid,
                multi_index_uniqgrid,
                ngamma_ref_grid,
                n_Texp_grid,
                qt,
                self.Tref_broadening,
                self.Twt,
            )
        else:
            raise ValueError("nstitch should be integer and larger than 1.")

        return xsv

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

        (
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        ) = self.opainfo

        if self.mdb.dbtype == "hitran":
            qtarr = vmap(self.mdb.qr_interp, (None, 0, None))(
                self.mdb.isotope, Tarr, self.Tref
            )
        elif self.mdb.dbtype == "exomol":
            qtarr = vmap(self.mdb.qr_interp, (0, None))(Tarr, self.Tref)

        if self.nstitch > 1:

            def floop(icarry, lbd_coeff):
                nu_grid_each = dynamic_slice(
                    self.nu_grid, (icarry * self.div_length,), (self.div_length,)
                )
                xsm_nu = self.xsmatrix_stitch[self.diffmode](
                    Tarr,
                    Parr,
                    self.Tref,
                    R,
                    lbd_coeff,
                    nu_grid_each,
                    ngamma_ref_grid,
                    n_Texp_grid,
                    multi_index_uniqgrid,
                    elower_grid,
                    self.mdb.molmass,
                    qtarr,
                    self.Tref_broadening,
                    self.filter_length_oneside,
                    self.Twt,
                )

                return icarry + 1, xsm_nu

            _, xsm_matrix = scan(floop, 0, self.lbd_coeff_reshaped)
            xsm_matrix = xsm_matrix / self.nu_grid_extended_array[:, jnp.newaxis, :]
            xsmatrix_ola_stitch = overlap_and_add_matrix(
                xsm_matrix, self.output_length, self.div_length
            )
            return xsmatrix_ola_stitch[
                :, self.filter_length_oneside : -self.filter_length_oneside
            ]

        elif self.nstitch == 1:
            xsmatrix_func = self.xsmatrix_close[self.diffmode]
            xsm = xsmatrix_func(
                Tarr,
                Parr,
                self.Tref,
                R,
                pmarray,
                self.lbd_coeff,
                self.nu_grid,
                ngamma_ref_grid,
                n_Texp_grid,
                multi_index_uniqgrid,
                elower_grid,
                self.mdb.molmass,
                qtarr,
                self.Tref_broadening,
                self.Twt,
            )
        else:
            raise ValueError("nstitch should be integer and larger than 1.")
        return xsm

    def plot_broadening_parameters(self, figname="broadpar_grid.png", crit=300000):
        """plot broadening parameters and grids

        Args:
            figname (str, optional): output image file. Defaults to "broadpar_grid.png".
            crit (int, optional): sampling criterion. Defaults to 300000. when the number of lines is huge and if it exceeded ~ crit, we sample the lines to reduce the computation.
        """
        from exojax.plot.opaplot import plot_broadening_parameters_grids

        _, _, ngamma_ref_grid, n_Texp_grid, _, _ = self.opainfo
        gamma_ref_in = self.gamma_ref
        n_Texp_in = self.n_Texp
        plot_broadening_parameters_grids(
            ngamma_ref_grid,
            n_Texp_grid,
            self.nu_grid,
            self.resolution,
            gamma_ref_in,
            n_Texp_in,
            crit,
            figname,
        )


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
        from exojax.spec import normalized_doppler_sigma, gamma_natural
        from exojax.spec.hitran import line_strength
        from exojax.spec.exomol import gamma_exomol
        from exojax.spec.hitran import gamma_hitran
        from exojax.spec.set_ditgrid import ditgrid_log_interval
        from exojax.spec.modit import xsvector_zeroscan
        from exojax.spec.modit import xsvector_open_zeroscan
        from exojax.spec import normalized_doppler_sigma

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
        from exojax.spec.modit import xsmatrix_zeroscan
        from exojax.spec.modit import xsmatrix_open_zeroscan
        from exojax.spec.modit import exomol
        from exojax.spec.modit import hitran

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


class OpaDirect(OpaCalc):
    def __init__(self, mdb, nu_grid, wavelength_order="descending"):
        """initialization of OpaDirect (LPF)



        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
        """
        super().__init__(nu_grid)

        # default setting
        self.method = "lpf"
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.mdb = mdb
        self.apply_params()

    def __eq__(self, other):
        """eq method for OpaDirect, definied by comparing all the attributes and important status

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, OpaDirect):
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
            qt = self.mdb.qr_interp(self.mdb.isotope, T, Tref_original)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air, self.mdb.gamma_self
            ) + gamma_natural(self.mdb.A)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T, Tref_original)
            gammaL = gamma_exomol(
                P, T, self.mdb.n_Texp, self.mdb.alpha_ref
            ) + gamma_natural(self.mdb.A)
        sigmaD = doppler_sigma(self.mdb.nu_lines, T, self.mdb.molmass)
        Sij = line_strength(
            T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt, Tref_original
        )
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
        from exojax.spec.atomll import gamma_vald3, interp_QT_284
        from exojax.spec.lpf import xsmatrix as xsmatrix_lpf

        numatrix = self.opainfo
        vmaplinestrengh = jit(vmap(line_strength, (0, None, None, None, 0, None)))
        if self.mdb.dbtype == "hitran":
            vmapqt = vmap(self.mdb.qr_interp, (None, 0, None))
            qt = vmapqt(self.mdb.isotope, Tarr, Tref_original)
            vmaphitran = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))
            gammaLM = vmaphitran(
                Parr,
                Tarr,
                np.zeros_like(Parr),
                self.mdb.n_air,
                self.mdb.gamma_air,
                self.mdb.gamma_self,
            ) + gamma_natural(self.mdb.A)
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif self.mdb.dbtype == "exomol":
            vmapqt = vmap(self.mdb.qr_interp, (0, None))
            qt = vmapqt(Tarr, Tref_original)
            vmapexomol = jit(vmap(gamma_exomol, (0, 0, None, None)))
            gammaLMP = vmapexomol(Parr, Tarr, self.mdb.n_Texp, self.mdb.alpha_ref)
            gammaLMN = gamma_natural(self.mdb.A)
            gammaLM = gammaLMP + gammaLMN[None, :]
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif (self.mdb.dbtype == "kurucz") or (self.mdb.dbtype == "vald"):
            qt_284 = vmap(interp_QT_284, (0, None, None))(
                Tarr, self.mdb.T_gQT, self.mdb.gQT_284species
            )
            qt_K = qt_284[:, self.mdb.QTmask]  # e.g., qt_284[:,76] #Fe I
            qr_K = qt_K / self.mdb.QTref_284[self.mdb.QTmask]
            vmapvald3 = jit(
                vmap(
                    gamma_vald3,
                    (
                        0,
                        0,
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
            )
            PH, PHe, PHH = (
                Parr * self.mdb.vmrH,
                Parr * self.mdb.vmrHe,
                Parr * self.mdb.vmrHH,
            )
            gammaLM = vmapvald3(
                Tarr,
                PH,
                PHH,
                PHe,
                self.mdb.ielem,
                self.mdb.iion,
                self.mdb.dev_nu_lines,
                self.mdb.elower,
                self.mdb.eupper,
                self.mdb.atomicmass,
                self.mdb.ionE,
                self.mdb.gamRad,
                self.mdb.gamSta,
                self.mdb.vdWdamp,
                1.0,
            )
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qr_K.T,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.atomicmass
            )

        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)
