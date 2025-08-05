"""API for Modified Discrete Integral Transform (MODIT) opacity calculations.

This module provides the OpaModit class for efficient opacity calculations
using the MODIT method, which provides a balance between accuracy and speed.
"""

import warnings
from typing import Optional, Union, Literal, List

import jax.numpy as jnp
import numpy as np

from exojax.opacity.base import OpaCalc
from exojax.opacity import initspec
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav
from exojax.utils.jaxstatus import check_jax64bit
from exojax.utils.instfunc import resolution_eslog
from exojax.opacity.modit.core import _setdgm


class OpaModit(OpaCalc):
    """Opacity Calculator Class for Modified Discrete Integral Transform (MODIT).

    MODIT provides a balance between accuracy and computational efficiency by using
    pre-computed discrete integral transforms with optimized grids.

    Attributes:
        method: Always "modit" for this calculator
        mdb: Molecular database instance
        wavelength_order: Order of wavelength grid
        opainfo: Opacity information (cont_nu, index_nu, R, pmarray)
        dgm_ngammaL: DIT grid matrix for gammaL (computed if Tarr_list/Parr provided)
        alias: Aliasing mode ("close" or "open")
        cutwing: Wing cut parameter for convolution
    """

    def __init__(
        self,
        mdb,
        nu_grid: Union[np.ndarray, jnp.ndarray],
        Tarr_list: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        Parr: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        Pself_ref: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        dit_grid_resolution: float = 0.2,
        allow_32bit: bool = False,
        alias: Literal["close", "open"] = "close",
        cutwing: float = 1.0,
        wavelength_order: Literal["ascending", "descending"] = "descending",
    ) -> None:
        """Initialize OpaModit opacity calculator.

        Note:
            Tarr_list and Parr are used to compute xsmatrix. Not required for xsvector.

        Args:
            mdb: Molecular database (mdbExomol, mdbHitemp, mdbHitran)
            nu_grid: Wavenumber grid in cm⁻¹
            Tarr_list: Temperature array(s) to be tested. Can be 1D or list of 1D arrays
                      such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr: Pressure array in bar
            Pself_ref: Self-pressure array in bar. If None, defaults to zero
            dit_grid_resolution: DIT grid resolution
            allow_32bit: If True, allow 32-bit mode of JAX
            alias: Aliasing mode - "open" gives open-type cross-section with aliasing parts,
                  "close" gives closed-type without aliasing
            cutwing: Wing cut for convolution in open cross section. Always 1.0 for "close" mode
            wavelength_order: Wavelength grid order

        Raises:
            ValueError: If gpu_transfer is False in mdb (required for MODIT)
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
            _, _, R, _ = self.opainfo
            self.dgm_ngammaL = _setdgm(
                self.mdb,
                self.dit_grid_resolution,
                R,
                Tarr_list,
                Parr,
                Pself_ref=Pself_ref,
            )
        else:
            warnings.warn("Tarr_list/Parr are needed for xsmatrix.", UserWarning)
        self.alias = alias
        self.cutwing = cutwing
        self.set_aliasing()

    def __eq__(self, other: object) -> bool:
        """Check equality with another OpaModit instance.

        Args:
            other: Object to compare with

        Returns:
            True if instances are equivalent, False otherwise
        """
        if not isinstance(other, OpaModit):
            return False

        return (
            (self.mdb == other.mdb)
            and (self.wavelength_order == other.wavelength_order)
            and np.array_equal(self.nu_grid, other.nu_grid)
        )

    def __ne__(self, other: object) -> bool:
        """Check inequality with another OpaModit instance."""
        return not self.__eq__(other)

    def apply_params(self) -> None:
        """Apply database parameters and initialize opacity info."""
        self.dbtype = self.mdb.dbtype
        self.opainfo = initspec.init_modit(self.mdb.nu_lines, self.nu_grid)
        self.ready = True

    def xsvector(self, T: float, P: float, Pself: float = 0.0) -> jnp.ndarray:
        """Compute cross section vector for given temperature and pressure.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar
            Pself: Self-pressure for HITEMP/HITRAN in bar

        Returns:
            Cross section vector in cm²
        """
        from exojax.database.core.broadening import gamma_natural, normalized_doppler_sigma
        from exojax.database.core.broadening import gamma_exomol
        from exojax.database.core.broadening import gamma_hitran
        from exojax.database.core.line_strength import line_strength
        from exojax.opacity.modit.modit import xsvector_open_zeroscan, xsvector_zeroscan
        from exojax.opacity._common.set_ditgrid import ditgrid_log_interval

        cont_nu, index_nu, R, pmarray = self.opainfo
        dbtype = self.mdb.dbtype

        # Compute broadening and line strength based on database type
        if dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, Tref_original)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air, self.mdb.gamma_self
            ) + gamma_natural(self.mdb.A)
        elif dbtype == "exomol":
            qt = self.mdb.qr_interp(T, Tref_original)
            gammaL = gamma_exomol(
                P, T, self.mdb.n_Texp, self.mdb.alpha_ref
            ) + gamma_natural(self.mdb.A)
        else:
            raise ValueError(
                f"Unsupported database type for xsvector: '{dbtype}'. "
                "Supported types: hitran, exomol"
            )
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

    def xsmatrix(
        self, Tarr: Union[np.ndarray, jnp.ndarray], Parr: Union[np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute cross section matrix for temperature and pressure arrays.

        Note:
            Self-pressure (Pself) is currently set to zero for HITEMP/HITRAN.

        Args:
            Tarr: Temperature array in K
            Parr: Pressure array in bar

        Returns:
            Cross section matrix with shape (Nlayer, N_wavenumber) in cm²

        Raises:
            ValueError: If database type is not supported
        """
        from exojax.opacity.modit.modit import (
            exomol,
            hitran,
            xsmatrix_open_zeroscan,
            xsmatrix_zeroscan,
        )

        cont_nu, index_nu, R, pmarray = self.opainfo
        dbtype = self.mdb.dbtype

        # Compute parameters based on database type
        if dbtype == "hitran":
            SijM, ngammaLM, nsigmaDl = hitran(
                self.mdb, Tarr, Parr, np.zeros_like(Parr), R, self.mdb.molmass
            )
        elif dbtype == "exomol":
            SijM, ngammaLM, nsigmaDl = exomol(self.mdb, Tarr, Parr, R, self.mdb.molmass)
        else:
            raise ValueError(
                f"Unsupported database type for xsmatrix: '{dbtype}'. "
                "Supported types: hitran, exomol"
            )
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
