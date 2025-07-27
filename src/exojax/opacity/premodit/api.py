"""API for Pre-computed Modified Discrete Integral Transform (PreMODIT) opacity calculations.

This module provides the OpaPremodit class for high-performance opacity calculations
using pre-computed grids. PreMODIT offers the fastest computation speed by using
optimized parameter grids and efficient memory management.
"""

from functools import partial
from typing import Optional, Union, Literal, Dict, Any, Tuple, List

import jax.numpy as jnp
import numpy as np
from jax import checkpoint, checkpoint_policies, vmap
from jax.lax import dynamic_slice, scan

from exojax.opacity.base import OpaCalc
from exojax.signal.ola import overlap_and_add, overlap_and_add_matrix
from exojax.opacity import initspec
from exojax.opacity.premodit.lbderror import optimal_params
from exojax.utils.checkarray import is_outside_range
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav, wavenumber_grid
from exojax.utils.instfunc import nx_even_from_resolution_eslog, resolution_eslog
from exojax.utils.jaxstatus import check_jax64bit

from exojax.opacity.premodit.core import _select_broadening_mode
from exojax.opacity.premodit.core import _compute_common_broadening_parameters


class OpaPremodit(OpaCalc):
    """Opacity Calculator Class for Pre-computed Modified Discrete Integral Transform (PreMODIT).

    PreMODIT provides the fastest opacity calculations by using pre-computed parameter grids
    and optimized memory management. It achieves high performance through efficient grid
    interpolation and reduced computational overhead.

    Attributes:
        method: Always "premodit" for this calculator
        mdb: Molecular database instance
        wavelength_order: Order of wavelength grid
        opainfo: Pre-computed opacity information and grids
        diffmode: Differentiation mode for optimization
        broadening_parameter_resolution: Broadening parameter resolution configuration
        single_broadening: Whether using single broadening mode
        ngrid_broadpar: Number of broadening parameter grid points
    """

    def __init__(
        self,
        mdb,
        nu_grid: Union[np.ndarray, jnp.ndarray],
        diffmode: int = 0,
        broadening_resolution: Dict[str, Any] = {"mode": "manual", "value": 0.2},
        auto_trange: Optional[Tuple[float, float]] = None,
        manual_params: Optional[Tuple[float, float, float]] = None,
        dit_grid_resolution: Optional[float] = None,
        allow_32bit: bool = False,
        nstitch: int = 1,
        cutwing: float = 1.0,
        wavelength_order: Literal["ascending", "descending"] = "descending",
        version_auto_trange: int = 2,
    ) -> None:
        """Initialize OpaPremodit opacity calculator.

        Note:
            If neither auto_trange nor manual_params are provided, use manual_setting()
            or provide self.dE, self.Twt and call self.apply_params().

        Args:
            mdb: Molecular database (mdbExomol, mdbHitemp, mdbHitran)
            nu_grid: Wavenumber grid in cm⁻¹
            diffmode: Differentiation mode for optimization
            broadening_resolution: Broadening parameter resolution configuration.
                - "manual": Use specified resolution value (higher memory usage)
                - "minmax": Use min/max values from database (medium memory)
                - "single": Use single broadening parameters (lowest memory)
            auto_trange: Temperature range [Tl, Tu] for 1% line strength precision
            manual_params: Manual PreMODIT parameter set [dE, Tref, Twt]
            dit_grid_resolution: Deprecated - use broadening_resolution instead
            allow_32bit: If True, allow 32-bit mode of JAX
            nstitch: Number of frequency domain stitching segments
            cutwing: Wing cut for convolution when nstitch > 1
            wavelength_order: Wavelength grid order
            version_auto_trange: Version of default elower grid trange file

        Raises:
            ValueError: If no molecular lines are within the wavenumber grid
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

        (
            self.broadening_parameter_resolution,
            self.dit_grid_resolution,
            self.single_broadening,
            self.single_broadening_parameters,
        ) = _select_broadening_mode(broadening_resolution, dit_grid_resolution)

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

    def __eq__(self, other: object) -> bool:
        """Check equality with another OpaPremodit instance.

        Args:
            other: Object to compare with

        Returns:
            True if instances are equivalent, False otherwise
        """
        if not isinstance(other, OpaPremodit):
            return False

        eq_attributes = (
            (self.mdb == other.mdb)
            and (self.diffmode == other.diffmode)
            and (self.ngrid_broadpar == other.ngrid_broadpar)
            and (self.wavelength_order == other.wavelength_order)
            and (self.version_auto_trange == other.version_auto_trange)
            and np.array_equal(self.nu_grid, other.nu_grid)
        )
        eq_attributes = self._if_exist_check_eq(other, "dE", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tref", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Twt", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tmax", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tmin", eq_attributes)
        eq_attributes = self._if_exist_check_eq(other, "Tref_broadening", eq_attributes)

        return eq_attributes

    def _if_exist_check_eq(
        self, other: object, attribute: str, eq_attributes: bool
    ) -> bool:
        """Check equality of optional attributes if they exist."""
        if hasattr(self, attribute) and hasattr(other, attribute):
            return eq_attributes and getattr(self, attribute) == getattr(
                other, attribute
            )
        elif not hasattr(self, attribute) and not hasattr(other, attribute):
            return eq_attributes
        else:
            return False

    def __ne__(self, other: object) -> bool:
        """Check inequality with another OpaPremodit instance."""
        return not self.__eq__(other)

    def auto_setting(self, Tl: float, Tu: float) -> None:
        """Automatically set PreMODIT parameters for given temperature range.

        Args:
            Tl: Lower temperature limit in K
            Tu: Upper temperature limit in K
        """
        print("OpaPremodit: params automatically set.")
        self.dE, self.Tref, self.Twt = optimal_params(
            Tl, Tu, self.diffmode, self.version_auto_trange
        )
        self.Tmax = Tu
        self.Tmin = Tl
        self.apply_params()

    def manual_setting(
        self,
        dE: float,
        Tref: float,
        Twt: float,
        Tmax: Optional[float] = None,
        Tmin: Optional[float] = None,
    ) -> None:
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

    def set_Tref_broadening_to_midpoint(self) -> None:
        """Set self.Tref_broadening using log midpoint of Tmax and Tmin."""
        from exojax.opacity.premodit.premodit import (
            reference_temperature_broadening_at_midpoint,
        )

        self.Tref_broadening = reference_temperature_broadening_at_midpoint(
            self.Tmin, self.Tmax
        )
        print("OpaPremodit: Tref_broadening is set to ", self.Tref_broadening, "K")

    def apply_params(self) -> None:
        """Apply parameters to the class and compute pre-computed grids.

        Defines self.lbd_coeff and self.opainfo for opacity calculations.
        """
        # self.mdb.change_reference_temperature(self.Tref)
        self.dbtype = self.mdb.dbtype

        # sets the broadening reference temperature
        if self.single_broadening:
            print("OpaPremodit: a single broadening parameter set is used.")
            self.Tref_broadening = Tref_original
        else:
            self.set_Tref_broadening_to_midpoint()

        # self.n_Texp, self.gamma_ref are defined with the reference temperature of Tref_broadening
        self.n_Texp, self.gamma_ref = _compute_common_broadening_parameters(
            self.mdb, self.Tref_broadening
        )

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
        from exojax.opacity.premodit.premodit import (
            xsmatrix_first,
            xsmatrix_nu_open_first,
            xsmatrix_nu_open_second,
            xsmatrix_nu_open_zeroth,
            xsmatrix_second,
            xsmatrix_zeroth,
            xsvector_first,
            xsvector_nu_open_first,
            xsvector_nu_open_second,
            xsvector_nu_open_zeroth,
            xsvector_second,
            xsvector_zeroth,
        )

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

    def xsvector(self, T: float, P: float) -> jnp.ndarray:
        """Compute cross section vector for given temperature and pressure.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar

        Returns:
            Cross section vector in cm²
        """
        from exojax.database.hitran import normalized_doppler_sigma

        (
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        ) = self.opainfo
        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, R)

        dbtype = self.mdb.dbtype
        if dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, self.Tref)
        elif dbtype == "exomol":
            qt = self.mdb.qr_interp(T, self.Tref)
        else:
            raise ValueError(
                f"Unsupported database type for xsvector: '{dbtype}'. "
                "Supported types: hitran, exomol"
            )

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

    def xsmatrix(
        self, Tarr: Union[np.ndarray, jnp.ndarray], Parr: Union[np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute cross section matrix for temperature and pressure arrays.

        Args:
            Tarr: Temperature array in K
            Parr: Pressure array in bar

        Returns:
            Cross section matrix with shape (Nlayer, N_wavenumber) in cm²

        Raises:
            ValueError: If nstitch configuration is invalid
        """

        (
            multi_index_uniqgrid,
            elower_grid,
            ngamma_ref_grid,
            n_Texp_grid,
            R,
            pmarray,
        ) = self.opainfo

        dbtype = self.mdb.dbtype
        if dbtype == "hitran":
            qtarr = vmap(self.mdb.qr_interp, (None, 0, None))(
                self.mdb.isotope, Tarr, self.Tref
            )
        elif dbtype == "exomol":
            qtarr = vmap(self.mdb.qr_interp, (0, None))(Tarr, self.Tref)
        else:
            raise ValueError(
                f"Unsupported database type for xsmatrix: '{dbtype}'. "
                "Supported types: hitran, exomol"
            )

        if self.nstitch > 1:

            @partial(
                checkpoint, policy=checkpoint_policies.dots_with_no_batch_dims_saveable
            )
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
