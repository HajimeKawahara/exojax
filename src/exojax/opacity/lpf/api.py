"""API for Line Profile Function (LPF) opacity calculations.

This module provides the OpaDirect class for direct line-by-line opacity
calculations using the LPF method.
"""

from typing import Callable, Literal, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exojax.opacity import initspec
from exojax.opacity.base import OpaCalc
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav


class OpaDirect(OpaCalc):
    """Opacity Calculator Class for Direct Line-by-Line calculations (LPF).

    This class directly sums the selected line profiles. The default is Voigt;
    ``alkali_subvoigt`` selects the Na/K-specific core and wing prescription.

    Attributes:
        method: Always "lpf" for this calculator
        mdb: Molecular or atomic line database instance
        wavelength_order: Order of wavelength grid
        line_profile: "voigt" or "alkali_subvoigt"
        opainfo: Opacity information from initialization

    Notes:
        VALD and Kurucz support both ``xsvector`` and ``xsmatrix``. Their
        H, He, and H2 partial pressures use the database's ``vmr_fraction``
        in that order. Select a single atomic species before applying its
        abundance to the cross section.

        NIST requires ``atomic_broadening`` because its line list does not
        provide damping parameters. A supplied callback replaces the total
        Lorentzian width for NIST, VALD, or Kurucz.
    """

    def __init__(
        self,
        mdb,
        nu_grid: np.ndarray,
        wavelength_order: Literal["ascending", "descending"] = "descending",
        *,
        line_profile: Literal["voigt", "alkali_subvoigt"] = "voigt",
        atomic_broadening: Optional[Callable] = None,
    ) -> None:
        """Initialize OpaDirect (LPF) opacity calculator.

        Args:
            mdb: Molecular or atomic line database
            nu_grid: Wavenumber grid in cm⁻¹
            wavelength_order: Order of wavelength grid
            line_profile: "voigt" (default) or "alkali_subvoigt". The latter
                requires a VALD or Kurucz selection containing only Na I or
                only K I and applies sub-Voigt wings to every selected line.
                Include line centers up to 9000 cm-1 outside the grid.
            atomic_broadening: JAX-compatible callable ``(T, P) -> gammaL``
                for NIST, VALD, or Kurucz. T is in K and P in bar. Return
                the total Lorentzian HWHM in cm-1, shaped ``(Nline,)``.
                Include every desired broadening contribution; no natural
                or pressure width is added automatically. Required for NIST.
        """
        if atomic_broadening is not None:
            if mdb.dbtype not in ("nist", "vald", "kurucz"):
                raise ValueError("atomic_broadening supports only NIST, VALD, and Kurucz.")
            if not callable(atomic_broadening):
                raise TypeError("atomic_broadening must be callable.")
        if mdb.dbtype == "nist" and atomic_broadening is None:
            raise ValueError("NIST requires an explicit atomic_broadening(T, P) callable.")
        super().__init__(nu_grid)

        self.method = "lpf"
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.mdb = mdb
        self.line_profile = line_profile
        self.atomic_broadening = atomic_broadening
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
            and (self.line_profile == other.line_profile)
            and (
                getattr(self, "atomic_broadening", None)
                is getattr(other, "atomic_broadening", None)
            )
            and (self.wavelength_order == other.wavelength_order)
            and np.array_equal(self.nu_grid, other.nu_grid)
        )

    def __ne__(self, other: object) -> bool:
        """Check inequality with another OpaDirect instance."""
        return not self.__eq__(other)

    def apply_params(self) -> None:
        """Apply database parameters and initialize opacity info."""
        self.dbtype = self.mdb.dbtype
        self._init_line_profile()
        self.opainfo = initspec.init_lpf(self.mdb.nu_lines, self.nu_grid)
        self._init_xsmatrix_wrappers()
        self.ready = True

    def _init_line_profile(self) -> None:
        """Validate the selected profile and set species-specific constants."""
        if self.line_profile not in ("voigt", "alkali_subvoigt"):
            raise ValueError("line_profile must be 'voigt' or 'alkali_subvoigt'.")
        if self.line_profile == "voigt":
            return
        if self.dbtype not in ("vald", "kurucz"):
            raise ValueError("alkali_subvoigt requires a VALD or Kurucz database.")
        elements = np.asarray(self.mdb._ielem)
        ions = np.asarray(self.mdb._iion)
        if (
            elements.size == 0
            or not np.all(ions == 1)
            or not (np.all(elements == 11) or np.all(elements == 19))
        ):
            raise ValueError("Select a single neutral species, Na I or K I, for alkali_subvoigt.")
        self.species = "Na" if elements[0] == 11 else "K"
        self.detuning_ref, self.wing_cutoff = (
            (30.0, 5000.0) if self.species == "Na" else (20.0, 1600.0)
        )

    def _init_xsmatrix_wrappers(self) -> None:
        """Build reusable JAX wrappers once per OpaDirect instance.

        Recreating ``jit(vmap(...))`` wrappers inside ``xsmatrix`` causes new
        Python function objects to appear on each call, which can interfere with
        JAX cache reuse even when shapes stay the same.
        """
        from exojax.database.core.broadening import doppler_sigma
        from exojax.database.core.broadening import gamma_exomol
        from exojax.database.core.broadening import gamma_hitran
        from exojax.database.core.line_strength import line_strength

        self._vmap_line_strength = jit(
            vmap(line_strength, (0, None, None, None, 0, None))
        )
        self._vmap_doppler_sigma = jit(vmap(doppler_sigma, (None, 0, None)))

        if self.dbtype == "hitran":
            self._vmap_qt = vmap(self.mdb.qr_interp_lines, (0, None))
            self._vmap_gamma = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))
        elif self.dbtype == "exomol":
            self._vmap_qt = vmap(self.mdb.qr_interp, (0, None))
            self._vmap_gamma = jit(vmap(gamma_exomol, (0, 0, None, None)))
        elif getattr(self, "atomic_broadening", None) is not None:
            self._vmap_qt = vmap(self.mdb.qr_interp_lines, (0, None))
            self._vmap_gamma = jit(vmap(self._atomic_gamma, (0, 0)))
        else:
            self._vmap_qt = None
            self._vmap_gamma = None

        if self.line_profile == "alkali_subvoigt":
            from exojax.opacity.alkali import _xsvector

            self._vmap_subvoigt = vmap(_xsvector, (None, 0, 0, 0, 0, None, None))

    def _atomic_gamma(self, T, P):
        """Check the static shape of a user-supplied atomic Lorentz width."""
        gammaL = jnp.asarray(self.atomic_broadening(T, P))
        expected_shape = (len(self.mdb.nu_lines),)
        if gammaL.shape != expected_shape:
            raise ValueError(
                f"atomic_broadening must return shape {expected_shape}, "
                f"but returned {gammaL.shape}."
            )
        return gammaL

    def _atomic_line_parameters(self, T, P):
        """Use the same atomic parameter calculation for vectors and matrices."""
        from exojax.opacity.lpf.lpf import vald

        Tarr = jnp.atleast_1d(T)
        Parr = jnp.atleast_1d(P)
        if getattr(self, "atomic_broadening", None) is not None:
            qr = self._vmap_qt(Tarr, self.mdb.Tref)
            SijM = self._vmap_line_strength(
                Tarr, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower,
                qr, self.mdb.Tref,
            )
            gammaLM = self._vmap_gamma(Tarr, Parr)
            sigmaDM = self._vmap_doppler_sigma(
                self.mdb.nu_lines, Tarr, self.mdb.line_masses
            )
            return SijM, gammaLM, sigmaDM
        return vald(
            self.mdb, Tarr, Parr * self.mdb.vmrH,
            Parr * self.mdb.vmrHe, Parr * self.mdb.vmrHH,
        )

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

        if dbtype == "hitran":
            qt = self.mdb.qr_interp_lines(T, Tref_original)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air, self.mdb.gamma_self
            ) + gamma_natural(self.mdb.A)
            line_masses = self.mdb.molmass
        elif dbtype == "exomol":
            qt = self.mdb.qr_interp(T, Tref_original)
            gammaL = gamma_exomol(
                P, T, self.mdb.n_Texp, self.mdb.alpha_ref
            ) + gamma_natural(self.mdb.A)
            line_masses = self.mdb.molmass
        elif dbtype == "hydrogen":
            qt = self.mdb.qr_interp_lines(T, Tref_original)
            gammaL = gamma_natural(self.mdb.A)
            line_masses = self.mdb.line_masses
        elif dbtype in ("kurucz", "vald", "nist"):
            SijM, gammaLM, sigmaDM = self._atomic_line_parameters(T, P)
            if self.line_profile == "alkali_subvoigt":
                from exojax.opacity.alkali import _xsvector

                return _xsvector(
                    numatrix, sigmaDM[0], gammaLM[0], SijM[0],
                    T, self.detuning_ref, self.wing_cutoff,
                )
            return xsvector_lpf(numatrix, sigmaDM[0], gammaLM[0], SijM[0])
        else:
            raise ValueError(
                f"Unsupported database type for xsvector: '{dbtype}'. "
                "Supported types: hitran, exomol, hydrogen, kurucz, vald, nist"
            )

        sigmaD = doppler_sigma(self.mdb.nu_lines, T, line_masses)
        Sij = line_strength(
            T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt, Tref_original
        )

        return xsvector_lpf(numatrix, sigmaD, gammaL, Sij)

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
        from exojax.database.core.broadening import gamma_natural
        from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf

        numatrix = self.opainfo
        dbtype = self.mdb.dbtype

        if dbtype == "hitran":
            qt = self._vmap_qt(Tarr, Tref_original)
            gammaLM = self._vmap_gamma(
                Parr,
                Tarr,
                jnp.zeros_like(Parr),
                self.mdb.n_air,
                self.mdb.gamma_air,
                self.mdb.gamma_self,
            ) + gamma_natural(self.mdb.A)
            SijM = self._vmap_line_strength(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = self._vmap_doppler_sigma(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif dbtype == "exomol":
            qt = self._vmap_qt(Tarr, Tref_original)
            gammaLMP = self._vmap_gamma(Parr, Tarr, self.mdb.n_Texp, self.mdb.alpha_ref)
            gammaLMN = gamma_natural(self.mdb.A)
            gammaLM = gammaLMP + gammaLMN[None, :]
            SijM = self._vmap_line_strength(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = self._vmap_doppler_sigma(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif dbtype == "hydrogen":
            qt = vmap(self.mdb.qr_interp_lines, (0, None))(Tarr, Tref_original)
            gammaLM = gamma_natural(self.mdb.A)[None, :] + jnp.zeros(
                (len(Tarr), 1)
            )
            SijM = self._vmap_line_strength(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = self._vmap_doppler_sigma(
                self.mdb.nu_lines, Tarr, self.mdb.line_masses
            )
        elif dbtype in ("kurucz", "vald", "nist"):
            SijM, gammaLM, sigmaDM = self._atomic_line_parameters(Tarr, Parr)
            if self.line_profile == "alkali_subvoigt":
                return self._vmap_subvoigt(
                    numatrix, sigmaDM, gammaLM, SijM,
                    Tarr, self.detuning_ref, self.wing_cutoff,
                )
        else:
            raise ValueError(
                f"Unsupported database type for xsmatrix: '{dbtype}'. "
                "Supported types: hitran, exomol, hydrogen, kurucz, vald, nist"
            )

        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)
