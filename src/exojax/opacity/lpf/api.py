"""API for Line Profile Function (LPF) opacity calculations.

This module provides the OpaDirect class for direct line-by-line opacity
calculations using the LPF method.
"""

from typing import Tuple, Union, Literal

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exojax.opacity.base import OpaCalc
from exojax.opacity import initspec
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav


class OpaDirect(OpaCalc):
    """Opacity Calculator Class for Direct Line-by-Line calculations (LPF).

    This class performs direct line-by-line opacity calculations without
    approximations, providing the most accurate results at the cost of
    computational efficiency.

    Attributes:
        method: Always "lpf" for this calculator
        mdb: Molecular database instance
        wavelength_order: Order of wavelength grid
        opainfo: Opacity information from initialization
    """

    def __init__(
        self,
        mdb,
        nu_grid: np.ndarray,
        wavelength_order: Literal["ascending", "descending"] = "descending",
    ) -> None:
        """Initialize OpaDirect (LPF) opacity calculator.

        Args:
            mdb: Molecular database (mdbExomol, mdbHitemp, mdbHitran, etc.)
            nu_grid: Wavenumber grid in cm⁻¹
            wavelength_order: Order of wavelength grid
        """
        super().__init__(nu_grid)

        # Configuration
        self.method = "lpf"
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.mdb = mdb
        self.apply_params()

    def __eq__(self, other: object) -> bool:
        """Check equality with another OpaDirect instance.

        Args:
            other: Object to compare with

        Returns:
            True if instances are equivalent, False otherwise
        """
        if not isinstance(other, OpaDirect):
            return False

        return (
            (self.mdb == other.mdb)
            and (self.wavelength_order == other.wavelength_order)
            and np.array_equal(self.nu_grid, other.nu_grid)
        )

    def __ne__(self, other: object) -> bool:
        """Check inequality with another OpaDirect instance."""
        return not self.__eq__(other)

    def apply_params(self) -> None:
        """Apply database parameters and initialize opacity info."""
        self.dbtype = self.mdb.dbtype
        self.opainfo = initspec.init_lpf(self.mdb.nu_lines, self.nu_grid)
        self.ready = True

    def xsvector(self, T: float, P: float, Pself: float = 0.0) -> jnp.ndarray:
        """Compute cross section vector for given temperature and pressure.

        Args:
            T: Temperature in Kelvin
            P: Pressure in bar
            Pself: Self-pressure for HITEMP/HITRAN in bar

        Returns:
            Cross section vector in cm²

        Raises:
            ValueError: If database type is not supported
        """
        from exojax.database.core.broadening import doppler_sigma, gamma_natural
        from exojax.database.core.broadening import gamma_exomol
        from exojax.database.core.broadening import gamma_hitran
        from exojax.database.core.line_strength import line_strength
        from exojax.opacity.lpf.lpf import xsvector as xsvector_lpf

        numatrix = self.opainfo
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

        # Compute Doppler broadening and line strength
        sigmaD = doppler_sigma(self.mdb.nu_lines, T, self.mdb.molmass)
        Sij = line_strength(
            T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt, Tref_original
        )

        return xsvector_lpf(numatrix, sigmaD, gammaL, Sij)

    def xsmatrix(self, Tarr: Union[np.ndarray, jnp.ndarray], Parr: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
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
        from exojax.database.core.broadening import doppler_sigma, gamma_natural
        from exojax.database.core_atom.broadening import gamma_vald3
        from exojax.database.core_atom.pf import interp_QT_284
        from exojax.database.core.broadening import gamma_exomol
        from exojax.database.core.broadening import gamma_hitran
        from exojax.database.core.line_strength import line_strength
        from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf

        numatrix = self.opainfo
        dbtype = self.mdb.dbtype
        
        # Vectorized line strength function (fixed typo)
        vmap_line_strength = jit(vmap(line_strength, (0, None, None, None, 0, None)))
        
        if dbtype == "hitran":
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
            SijM = vmap_line_strength(
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
        elif dbtype == "exomol":
            vmapqt = vmap(self.mdb.qr_interp, (0, None))
            qt = vmapqt(Tarr, Tref_original)
            vmapexomol = jit(vmap(gamma_exomol, (0, 0, None, None)))
            gammaLMP = vmapexomol(Parr, Tarr, self.mdb.n_Texp, self.mdb.alpha_ref)
            gammaLMN = gamma_natural(self.mdb.A)
            gammaLM = gammaLMP + gammaLMN[None, :]
            SijM = vmap_line_strength(
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
        elif dbtype in ("kurucz", "vald"):
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
            SijM = vmap_line_strength(
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
        else:
            raise ValueError(
                f"Unsupported database type for xsmatrix: '{dbtype}'. "
                "Supported types: hitran, exomol, kurucz, vald"
            )

        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)
