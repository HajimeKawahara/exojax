"""Continuum opacity calculator classes for ExoJAX.

This module provides opacity calculators for various continuum sources:
- CIA (Collision-Induced Absorption): Molecular collisions (H2-H2, H2-He, etc.)
- H-minus: Negative hydrogen ion continuum opacity
- Rayleigh: Molecular Rayleigh scattering
- Mie: Aerosol/cloud particle scattering using Mie theory

These classes compute continuum opacities that are added to line opacities
to provide complete atmospheric opacity for radiative transfer calculations.
None of these classes assume fixed T-P structure or atmospheric grids.
"""

import warnings
from typing import Union, Tuple, Optional
import logging

import jax.numpy as jnp
import numpy as np
from jax import vmap

from exojax.opacity.base import OpaCont
from exojax.database.core.abscoeff import interp_logacia_matrix
from exojax.database.core.abscoeff import interp_logacia_vector
from exojax.database.hminus import log_hminus_continuum
from exojax.database.mie import mie_lognormal_pymiescatt
from exojax.opacity.rayleigh import xsvector_rayleigh_gas

logger = logging.getLogger(__name__)

__all__ = ["OpaCIA", "OpaHminus", "OpaRayleigh", "OpaMie"]


class OpaCIA(OpaCont):
    """Opacity Continuum Calculator Class for Collision-Induced Absorption (CIA).
    
    Computes continuum opacity from molecular collisions such as H2-H2, H2-He,
    N2-N2, etc. Uses pre-computed CIA databases from HITRAN or other sources.
    
    Attributes:
        method: Always "cia" for this calculator
        cdb: CIA continuum database instance
        nu_grid: Wavenumber grid in cm⁻¹
        ready: Whether the calculator is ready for computation
    """

    def __init__(
        self, 
        cdb, 
        nu_grid: Union[np.ndarray, jnp.ndarray]
    ) -> None:
        """Initialize opacity calculator for CIA.

        Args:
            cdb: Continuum database containing CIA coefficients
            nu_grid: Wavenumber grid in cm⁻¹
            
        Raises:
            ValueError: If nu_grid is empty or invalid
        """
        if len(nu_grid) == 0:
            raise ValueError("nu_grid cannot be empty")
            
        self.method = "cia"
        self.warning = True
        self.nu_grid = nu_grid
        self.cdb = cdb
        self.ready = True

    def logacia_vector(self, T: float) -> jnp.ndarray:
        """Compute absorption coefficient vector of CIA.

        Args:
            T: Temperature in Kelvin

        Returns:
            Logarithm of absorption coefficient [Nnus] at T in units of cm⁵
        """
        return interp_logacia_vector(
            T, self.nu_grid, self.cdb.nucia, self.cdb.tcia, self.cdb.logac
        )

    def logacia_matrix(
        self, 
        temperatures: Union[np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute absorption coefficient matrix of CIA.

        Args:
            temperatures: Temperature array in Kelvin [Nlayer]

        Returns:
            Logarithm of absorption coefficient [Nlayer, Nnus] in units of cm⁵
        """
        return interp_logacia_matrix(
            temperatures, self.nu_grid, self.cdb.nucia, self.cdb.tcia, self.cdb.logac
        )


class OpaHminus(OpaCont):
    """Opacity Continuum Calculator Class for H-minus (H⁻) continuum.
    
    Computes continuum opacity from negative hydrogen ions (H⁻) which is
    important in stellar atmospheres and brown dwarf atmospheres. Includes
    both bound-free and free-free transitions.
    
    Attributes:
        method: Always "hminus" for this calculator
        nu_grid: Wavenumber grid in cm⁻¹
        ready: Whether the calculator is ready for computation
    """

    def __init__(self, nu_grid: Union[np.ndarray, jnp.ndarray]) -> None:
        """Initialize opacity calculator for H-minus continuum.
        
        Args:
            nu_grid: Wavenumber grid in cm⁻¹
            
        Raises:
            ValueError: If nu_grid is empty or invalid
        """
        if len(nu_grid) == 0:
            raise ValueError("nu_grid cannot be empty")
            
        self.method = "hminus"
        self.warning = True
        self.nu_grid = nu_grid
        self.ready = True

    def logahminus_matrix(
        self, 
        temperatures: Union[np.ndarray, jnp.ndarray], 
        number_density_e: Union[np.ndarray, jnp.ndarray], 
        number_density_h: Union[np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute absorption coefficient matrix of H⁻ continuum.

        Args:
            temperatures: Temperature array in Kelvin [Nlayer]
            number_density_e: Number density of electrons in cm⁻³ [Nlayer]
            number_density_h: Number density of hydrogen atoms in cm⁻³ [Nlayer]

        Returns:
            log10(absorption coefficient) in cm⁻¹ [Nlayer, Nnu]
        """
        return log_hminus_continuum(
            self.nu_grid, temperatures, number_density_e, number_density_h
        )


class OpaRayleigh(OpaCont):
    """Opacity Calculator for Rayleigh Scattering.
    
    Computes Rayleigh scattering cross sections for atmospheric gases.
    Uses molecular polarizability and King correction factors for accurate
    scattering calculations.
    
    Attributes:
        method: Always "rayleigh" for this calculator
        nu_grid: Wavenumber grid in cm⁻¹
        molname: Molecule name (e.g., "N2", "H2", "He")
        polarizability: Molecular polarizability
        king_factor: King correction factor for anisotropy
        ready: Whether the calculator is ready for computation
    """
    
    def __init__(
        self, 
        nu_grid: Union[np.ndarray, jnp.ndarray], 
        molname: str
    ) -> None:
        """Initialize Rayleigh scattering opacity calculator.

        Args:
            nu_grid: Wavenumber grid in cm⁻¹
            molname: Gas molecule name, such as "N2", "H2", "He"
            
        Raises:
            ValueError: If nu_grid is empty or molname is invalid
        """
        if len(nu_grid) == 0:
            raise ValueError("nu_grid cannot be empty")
        if not isinstance(molname, str) or len(molname.strip()) == 0:
            raise ValueError("molname must be a non-empty string")
            
        self.method = "rayleigh"
        self.nu_grid = nu_grid
        self.molname = molname.strip()
        self.set_auto_polarizability()
        self.set_auto_king_factor()
        self.check_ready()

    def set_auto_polarizability(self) -> None:
        """Automatically set molecular polarizability from database."""
        from exojax.atm.polarizability import polarizability

        try:
            self.polarizability = polarizability[self.molname]
        except KeyError:
            self.polarizability = None
            warnings.warn(
                f"No polarizability found for molecule '{self.molname}'. "
                "Set opa.polarizability manually.", UserWarning
            )

    def set_auto_king_factor(self) -> None:
        """Automatically set King correction factor from database."""
        from exojax.atm.polarizability import king_correction_factor

        try:
            self.king_factor = king_correction_factor[self.molname]
        except KeyError:
            self.king_factor = 1.0
            warnings.warn(
                f"No King correction factor found for molecule '{self.molname}'. "
                "Using default value of 1.0. Modify by setting opa.king_factor.", UserWarning
            )

    def check_ready(self) -> None:
        """Check if calculator is ready and update status."""
        if self.polarizability is None:
            logger.warning("No polarizability set. OpaRayleigh not ready for computation")
            self.ready = False
        else:
            logger.info("OpaRayleigh ready for computation")
            self.ready = True

    def xsvector(self) -> jnp.ndarray:
        """Compute cross section vector of Rayleigh scattering.

        Returns:
            Rayleigh scattering cross section vector [Nnus] in cm²
        """
        return xsvector_rayleigh_gas(
            self.nu_grid, self.polarizability, king_factor=self.king_factor
        )


class OpaMie(OpaCont):
    """Opacity Calculator for Mie Scattering from Aerosols and Clouds.
    
    Computes Mie scattering parameters for spherical particles using
    pre-computed grids or direct PyMieScatt calculations. Handles
    lognormal size distributions of condensate particles.
    
    Attributes:
        method: Always "mie" for this calculator
        nu_grid: Wavenumber grid in cm⁻¹
        pdb: Particle database with refractive indices
        ready: Whether the calculator is ready for computation
    """
    
    def __init__(
        self,
        pdb,
        nu_grid: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Initialize Mie scattering opacity calculator.
        
        Args:
            pdb: Particle database containing refractive indices
            nu_grid: Wavenumber grid in cm⁻¹
            
        Raises:
            ValueError: If nu_grid is empty or invalid
        """
        if len(nu_grid) == 0:
            raise ValueError("nu_grid cannot be empty")
            
        self.method = "mie"
        self.nu_grid = nu_grid
        self.pdb = pdb
        self.ready = True

    def mieparams_vector(
        self, 
        rg: float, 
        sigmag: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Interpolate Mie parameters vector from pre-computed grid.

        Args:
            rg: Geometric mean radius in lognormal distribution (cm)
            sigmag: Geometric standard deviation in lognormal distribution

        Returns:
            Tuple containing:
                - sigma_extinction: Extinction cross section (cm²)
                - sigma_scattering: Scattering cross section (cm²) 
                - asymmetric_factor: Mean asymmetry parameter g

        Notes:
            Based on Ackerman & Marley (2001) lognormal size distribution.
            Cross sections are normalized by reference number density N0.
        """

        # loads grid (mieparams in cgs)
        sigexg, sigscg, gg = (
            self.pdb.mieparams_cgs_at_refraction_index_wavenumber_from_miegrid(
                rg, sigmag
            )
        )

        # interpolation
        sigma_extinction = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, sigexg
        )
        sigma_scattering = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, sigscg
        )
        asymmetric_factor = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, gg
        )

        return sigma_extinction, sigma_scattering, asymmetric_factor

    def mieparams_matrix(
        self, 
        rg_layer: Union[np.ndarray, jnp.ndarray], 
        sigmag_layer: Union[np.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Interpolate Mie parameters matrix from pre-computed grid.
        
        Args:
            rg_layer: Geometric mean radius for each layer (cm) [Nlayer]
            sigmag_layer: Geometric standard deviation for each layer [Nlayer]

        Returns:
            Tuple containing:
                - sigma_extinction: Extinction cross section matrix (cm²) [Nlayer, Nnu]
                - sigma_scattering: Scattering cross section matrix (cm²) [Nlayer, Nnu]
                - asymmetric_factor: Asymmetry parameter matrix [Nlayer, Nnu]

        Notes:
            Uses vectorized computation for efficient layer-by-layer calculation.
            Cross sections are normalized by reference number density N0.
        """

        f = vmap(self.mieparams_vector, (0, 0), 0)
        return f(rg_layer, sigmag_layer)

    def mieparams_vector_direct_from_pymiescatt(
        self, 
        rg: float, 
        sigmag: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Mie parameters directly from PyMieScatt (slow but accurate).

        Args:
            rg: Geometric mean radius in lognormal distribution (cm)
            sigmag: Geometric standard deviation in lognormal distribution

        Returns:
            Tuple containing:
                - sigma_extinction: Extinction cross section (cm²)
                - sigma_scattering: Scattering cross section (cm²)
                - asymmetric_factor: Mean asymmetry parameter g

        Notes:
            Direct PyMieScatt calculation - no pre-computed grid needed.
            Slower than grid interpolation but more flexible for arbitrary parameters.
            Progress bar shows calculation status.
        """
        from tqdm import tqdm

        from exojax.database.mie import auto_rgrid

        # restrict wavenumber grid
        nind = len(self.pdb.refraction_index_wavenumber)
        numin = np.min(self.nu_grid)
        imin = np.searchsorted(self.pdb.refraction_index_wavenumber, numin)
        imin = np.max([imin - 1, 0])
        numax = np.max(self.nu_grid)
        imax = np.searchsorted(self.pdb.refraction_index_wavenumber, numax)
        imax = np.min([imax + 1, nind])

        refraction_index_wavenumber_restricted = self.pdb.refraction_index_wavenumber[
            imin:imax
        ]
        nind = len(refraction_index_wavenumber_restricted)
        refraction_index_restricted = self.pdb.refraction_index[imin:imax]

        # generates grid
        convfactor_to_cgs = (
            1.0e-8 / self.pdb.N0
        )  # conversion to cgs(1/Mega meter to 1/cm)

        sigexg = np.zeros(nind)
        sigscg = np.zeros(nind)
        gg = np.zeros(nind)

        cm2nm = 1.0e7
        rg_nm = rg * cm2nm
        rgrid = auto_rgrid(rg_nm, sigmag)
        for ind_m, m in enumerate(tqdm(refraction_index_restricted)):
            coeff = mie_lognormal_pymiescatt(
                m,
                refraction_index_wavenumber_restricted[ind_m],
                sigmag,
                rg_nm,
                self.pdb.N0,
                rgrid,
            )

            sigexg[ind_m] = coeff[0] * convfactor_to_cgs
            sigscg[ind_m] = coeff[1] * convfactor_to_cgs
            gg[ind_m] = coeff[3]

        # interpolation
        sigma_extinction = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, sigexg
        )
        sigma_scattering = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, sigscg
        )
        asymmetric_factor = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, gg
        )

        return sigma_extinction, sigma_scattering, asymmetric_factor

    def mieparams_matrix_direct_from_pymiescatt(
        self, 
        rg_layer: Union[np.ndarray, jnp.ndarray], 
        sigmag_layer: Union[np.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute Mie parameters matrix directly from PyMieScatt (slow).
        
        Args:
            rg_layer: Geometric mean radius for each layer (cm) [Nlayer]
            sigmag_layer: Geometric standard deviation for each layer [Nlayer]

        Returns:
            Tuple containing:
                - sigma_extinction: Extinction cross section matrix (cm²) [Nlayer, Nnu]
                - sigma_scattering: Scattering cross section matrix (cm²) [Nlayer, Nnu] 
                - asymmetric_factor: Asymmetry parameter matrix [Nlayer, Nnu]

        Notes:
            Uses vectorized direct PyMieScatt calculations.
            Slower than grid interpolation but avoids pre-computation requirements.
        """

        f = vmap(self.mieparams_vector_direct_from_pymiescatt, (0, 0), 0)
        return f(rg_layer, sigmag_layer)
