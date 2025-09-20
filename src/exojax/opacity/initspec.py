"""Initialization functions for opacity computation in ExoJAX.

This module provides initialization functions for different opacity calculation methods:
- LPF (Line Profile Function): Direct line-by-line calculations
- DIT (Discrete Integral Transform): Basic integral transform method
- MODIT (Modified DIT): Enhanced integral transform with optimization
- PreMODIT (Pre-computed MODIT): Fastest method using pre-computed grids

These functions prepare the necessary grids, matrices, and parameters required
for efficient opacity calculations across different temperature and pressure ranges.
"""

import warnings
from typing import Optional, Tuple, Union, List
import logging

import jax.numpy as jnp
import numpy as np

from exojax.opacity.lpf.make_numatrix import make_numatrix0
from exojax.opacity.premodit.premodit import (
    generate_lbd,
    make_broadpar_grid,
    make_elower_grid,
)
from exojax.utils.indexing import npgetix
from exojax.utils.instfunc import resolution_eslog

logger = logging.getLogger(__name__)


def init_lpf(
    nu_lines: Union[np.ndarray, jnp.ndarray], nu_grid: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Initialize LPF (Line Profile Function) opacity calculations.

    Args:
        nu_lines: Wavenumber list of lines [Nline] in cm⁻¹ (should be numpy F64)
        nu_grid: Wavenumber grid [Nnugrid] in cm⁻¹ (should be numpy F64)

    Returns:
        Wavenumber matrix [Nline, Nnu] for direct line-by-line calculations

    Raises:
        ValueError: If input arrays are empty or have incompatible shapes
    """
    if len(nu_lines) == 0:
        raise ValueError("nu_lines array is empty")
    if len(nu_grid) == 0:
        raise ValueError("nu_grid array is empty")

    numatrix = make_numatrix0(nu_grid, nu_lines, warning=True)
    return numatrix


def init_dit(
    nu_lines: Union[np.ndarray, jnp.ndarray],
    nu_grid: Union[np.ndarray, jnp.ndarray],
    warning: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Initialize DIT (Discrete Integral Transform) opacity calculations.

    Generates wavenumber contribution and index arrays for line shape density
    calculations using the discrete integral transform method.

    Args:
        nu_lines: Wavenumber list of lines [Nline] in cm⁻¹ (should be numpy F64)
        nu_grid: Wavenumber grid [Nnugrid] in cm⁻¹ (should be numpy F64)
        warning: If True, show dtype and range warnings

    Returns:
        Tuple containing:
            - cont: Contribution array for interpolation
            - index: Index array for grid mapping
            - pmarray: Alternating (+1, -1) array of length len(nu_grid)+1

    Note:
        cont is the contribution for i=index+1. (1 - cont) is the contribution
        for i=index. For other i, the contribution should be zero.
    """
    if len(nu_lines) == 0:
        raise ValueError("nu_lines array is empty")
    if len(nu_grid) == 0:
        raise ValueError("nu_grid array is empty")
        
    warn_dtype64(nu_lines, warning, tag="nu_lines")
    warn_dtype64(nu_grid, warning, tag="nu_grid")
    warn_outside_wavenumber_grid(nu_lines, nu_grid)
    warn_out_of_nu_grid(nu_lines, nu_grid)

    cont, index = npgetix(nu_lines, nu_grid)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0

    return jnp.array(cont), jnp.array(index), pmarray


def init_modit(
    nu_lines: Union[np.ndarray, jnp.ndarray],
    nu_grid: Union[np.ndarray, jnp.ndarray],
    warning: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
    """Initialize MODIT (Modified Discrete Integral Transform) opacity calculations.

    Generates wavenumber contribution and index arrays for the modified discrete
    integral transform method, which provides enhanced accuracy over basic DIT.

    Args:
        nu_lines: Wavenumber list of lines [Nline] in cm⁻¹ (should be numpy F64)
        nu_grid: Wavenumber grid [Nnugrid] in cm⁻¹ (should be numpy F64)
        warning: If True, show dtype and range warnings

    Returns:
        Tuple containing:
            - cont: Contribution array for interpolation
            - index: Index array for grid mapping
            - spectral_resolution: Spectral resolution (R)
            - pmarray: Alternating (+1, -1) array of length len(nu_grid)+1

    Note:
        cont is the contribution for i=index+1. (1 - cont) is the contribution
        for i=index. For other i, the contribution should be zero. Uses numpy
        for dq computation to avoid float32 truncation errors.
    """
    if len(nu_lines) == 0:
        raise ValueError("nu_lines array is empty")
    if len(nu_grid) == 0:
        raise ValueError("nu_grid array is empty")
        
    warn_dtype64(nu_lines, warning, tag="nu_lines")
    warn_dtype64(nu_grid, warning, tag="nu_grid")
    warn_outside_wavenumber_grid(nu_lines, nu_grid)
    warn_out_of_nu_grid(nu_lines, nu_grid)

    spectral_resolution = resolution_eslog(nu_grid)
    cont, index = npgetix(nu_lines, nu_grid)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0

    return jnp.array(cont), jnp.array(index), spectral_resolution, jnp.array(pmarray)


def warn_out_of_nu_grid(
    nu_lines: Union[np.ndarray, jnp.ndarray], nu_grid: Union[np.ndarray, jnp.ndarray]
) -> None:
    """Warning for line centers outside the wavenumber grid.

    Note:
        See Issue 341: https://github.com/HajimeKawahara/exojax/issues/341
        Only affects DIT and MODIT. For PreMODIT or newer OpacCalc,
        this issue is automatically fixed.

    Args:
        nu_lines: Line center wavenumbers in cm⁻¹
        nu_grid: Wavenumber grid in cm⁻¹
    """
    if nu_lines[0] < nu_grid[0] or nu_lines[-1] > nu_grid[-1]:
        warnings.warn("There are lines whose center is out of nu_grid", UserWarning)
        logger.warning("Lines outside nu_grid may cause edge artifacts. See Issue #341")
        logger.info("Issue URL: https://github.com/HajimeKawahara/exojax/issues/341")
        logger.info("Line center [cm⁻¹] vs nu_grid [cm⁻¹]:")
        logger.info("  Left: line=%.3f, grid=%.3f", nu_lines[0], nu_grid[0])
        logger.info("  Right: line=%.3f, grid=%.3f", nu_lines[-1], nu_grid[-1])


def init_premodit(
    nu_lines: Union[np.ndarray, jnp.ndarray],
    nu_grid: Union[np.ndarray, jnp.ndarray],
    elower: Union[np.ndarray, jnp.ndarray],
    gamma_ref: Union[np.ndarray, jnp.ndarray],
    n_Texp: Union[np.ndarray, jnp.ndarray],
    line_strength_ref: Union[np.ndarray, jnp.ndarray],
    Twt: float,
    Tref: float,
    Tref_broadening: float,
    Tmax: Optional[float] = None,
    Tmin: Optional[float] = None,
    dE: float = 160.0,
    dit_grid_resolution: float = 0.2,
    diffmode: int = 0,
    single_broadening: bool = False,
    single_broadening_parameters: Optional[List[Optional[float]]] = None,
    warning: bool = False,
) -> Tuple[
    jnp.ndarray,  # lbd_coeff
    jnp.ndarray,  # multi_index_uniqgrid
    jnp.ndarray,  # elower_grid
    jnp.ndarray,  # ngamma_ref_grid
    jnp.ndarray,  # n_Texp_grid
    float,  # R
    jnp.ndarray,  # pmarray
]:
    """Initialize PreMODIT (Pre-computed Modified DIT) opacity calculations.

    Sets up pre-computed grids and coefficients for the fastest opacity calculation
    method. PreMODIT achieves high performance through optimized parameter grids
    and efficient memory management.

    Args:
        nu_lines: Wavenumber list of lines [Nline] in cm⁻¹ (should be numpy F64)
        nu_grid: Wavenumber grid [Nnugrid] in cm⁻¹ (should be numpy F64)
        elower: Lower state energy of lines in cm⁻¹
        gamma_ref: Reference half-width (alpha_ref for ExoMol, gamma_air for HITRAN/HITEMP)
        n_Texp: Temperature exponent (n_Texp for ExoMol, n_air for HITRAN/HITEMP)
        line_strength_ref: Line strength at reference temperature Tref
        Twt: Temperature for weight in Kelvin
        Tref: Reference temperature for PreMODIT grid in Kelvin
        Tref_broadening: Reference temperature for broadening in Kelvin
        Tmax: Max temperature for n_Texp grid construction. If None, max(Twt, Tref) is used
        Tmin: Min temperature for n_Texp grid construction. If None, min(Twt, Tref) is used
        dE: Lower state energy grid interval in cm⁻¹
        dit_grid_resolution: DIT grid resolution. When np.inf, minmax simplex is used
        diffmode: Taylor expansion order for weight calculation (0, 1, or 2)
        single_broadening: If True, use single broadening parameters for all lines
        single_broadening_parameters: [gamma_ref, n_Texp] at 296K for single broadening mode
        warning: If True, show dtype and range warnings

    Returns:
        Tuple containing:
            - lbd_coeff: Pre-computed opacity coefficients
            - multi_index_uniqgrid: Multi-dimensional grid indices
            - elower_grid: Lower state energy grid
            - ngamma_ref_grid: Normalized reference width grid
            - n_Texp_grid: Temperature exponent grid
            - R: Spectral resolution
            - pmarray: Alternating (+1, -1) array of length len(nu_grid)+1

    Note:
        Uses numpy for computations to avoid float32 truncation errors.
        Only includes lines within the wavenumber grid to avoid edge artifacts.
    """
    # Input validation
    if len(nu_lines) == 0:
        raise ValueError("nu_lines array is empty")
    if len(nu_grid) == 0:
        raise ValueError("nu_grid array is empty")
    if len(elower) != len(nu_lines):
        raise ValueError(f"elower length ({len(elower)}) must match nu_lines length ({len(nu_lines)})")
    if len(gamma_ref) != len(nu_lines):
        raise ValueError(f"gamma_ref length ({len(gamma_ref)}) must match nu_lines length ({len(nu_lines)})")
    if len(n_Texp) != len(nu_lines):
        raise ValueError(f"n_Texp length ({len(n_Texp)}) must match nu_lines length ({len(nu_lines)})")
    if len(line_strength_ref) != len(nu_lines):
        raise ValueError(f"line_strength_ref length ({len(line_strength_ref)}) must match nu_lines length ({len(nu_lines)})")
    if Twt <= 0 or Tref <= 0 or Tref_broadening <= 0:
        raise ValueError("Temperatures must be positive")
    if diffmode not in [0, 1, 2]:
        raise ValueError("diffmode must be 0, 1, or 2")
        
    warn_dtype64(nu_lines, warning, tag="nu_lines")
    warn_dtype64(nu_grid, warning, tag="nu_grid")
    warn_dtype64(elower, warning, tag="elower")
    warn_outside_wavenumber_grid(nu_lines, nu_grid)

    if Tmax is None:
        Tmax = np.max([Twt, Tref])
    if Tmin is None:
        Tmin = np.min([Twt, Tref])

    R = resolution_eslog(nu_grid)
    ngamma_ref = gamma_ref / nu_lines * R
    elower_grid = make_elower_grid(elower, dE)

    if single_broadening:
        ngamma_ref_grid, n_Texp_grid = broadening_grid_for_single_broadening_mode(
            nu_lines, gamma_ref, n_Texp, single_broadening_parameters, R
        )
    else:
        ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
            ngamma_ref,
            n_Texp,
            Tmax,
            Tmin,
            Tref_broadening,
            dit_grid_resolution=dit_grid_resolution,
        )

    logger.info(
        "PreMODIT grid setup: %d reference width points, %d temperature exponent points",
        len(ngamma_ref_grid),
        len(n_Texp_grid),
    )
    wavmask = (nu_lines >= nu_grid[0]) * (nu_lines <= nu_grid[-1])  # Issue 341
    
    lbd_coeff, multi_index_uniqgrid = generate_lbd(
        line_strength_ref[wavmask],
        nu_lines[wavmask],
        nu_grid,
        ngamma_ref[wavmask],
        ngamma_ref_grid,
        n_Texp[wavmask],
        n_Texp_grid,
        elower[wavmask],
        elower_grid,
        Twt,
        Tref=Tref,
        diffmode=diffmode,
    )
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    pmarray = jnp.array(pmarray)

    return (
        lbd_coeff,
        multi_index_uniqgrid,
        elower_grid,
        ngamma_ref_grid,
        n_Texp_grid,
        R,
        pmarray,
    )


def broadening_grid_for_single_broadening_mode(
    nu_lines: Union[np.ndarray, jnp.ndarray],
    gamma_ref: Union[np.ndarray, jnp.ndarray],
    n_Texp: Union[np.ndarray, jnp.ndarray],
    single_broadening_parameters: List[Optional[float]],
    R: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create broadening parameter grids for single broadening mode.
    
    Args:
        nu_lines: Line center wavenumbers in cm⁻¹
        gamma_ref: Reference broadening parameters
        n_Texp: Temperature exponents
        single_broadening_parameters: [gamma_ref, n_Texp] values for single mode
        R: Spectral resolution
        
    Returns:
        Tuple of (normalized gamma_ref grid, n_Texp grid)
    """
    # Set normalized reference width
    if single_broadening_parameters[0] is not None:
        ngamma_ref_grid = jnp.array(
            [single_broadening_parameters[0] / np.median(nu_lines) * R]
        )
    else:
        ngamma_ref_grid = jnp.array([np.median(gamma_ref / nu_lines) * R])
    
    # Set temperature exponent
    if single_broadening_parameters[1] is not None:
        n_Texp_grid = jnp.array([single_broadening_parameters[1]])
    else:
        n_Texp_grid = jnp.array([np.median(n_Texp)])
        
    return ngamma_ref_grid, n_Texp_grid


def init_modit_vald(
    nu_linesM: Union[np.ndarray, jnp.ndarray],
    nus: Union[np.ndarray, jnp.ndarray],
    N_usp: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, np.ndarray]:
    """Initialize MODIT for atomic spectral databases from VALD.

    Special initialization function for VALD (Vienna Atomic Line Database)
    that handles multiple species with potentially missing lines.

    Args:
        nu_linesM: Wavenumbers of lines for each species [N_species x N_line] in cm⁻¹
        nus: Wavenumber grid [Nnugrid] in cm⁻¹ (should be numpy F64)
        N_usp: Number of atomic species

    Returns:
        Tuple containing:
            - contS: Contribution array [N_species x N_line]
            - indexS: Index array [N_species x N_line]
            - R: Spectral resolution
            - pmarray: Alternating (+1, -1) array of length len(nus)+1
    """
    contS = np.zeros_like(nu_linesM)
    indexS = np.zeros_like(contS)
    for i in range(N_usp):
        nu_lines = nu_linesM[i]
        nu_lines_nan = np.where(nu_lines == 0, np.nan, nu_lines)
        contS[i], indexnu_dammy, R, pmarray = init_modit(
            nu_lines_nan, nus
        )  # np.array(a), np.array(b), c, np.array(d)
        indexS[i] = np.hstack(
            [
                indexnu_dammy[np.where(~np.isnan(nu_lines_nan))],
                (len(nus) + 1)
                * np.ones(len(np.where(np.isnan(nu_lines_nan))[0]), dtype="int32"),
            ]
        )
    contS = jnp.array(contS)
    indexS = jnp.array(indexS, dtype="int32")
    return contS, indexS, R, pmarray


def warn_dtype64(
    arr: Union[np.ndarray, jnp.ndarray], warning: bool, tag: str = ""
) -> None:
    """Check array's data type and warn if not float64.

    Args:
        arr: Input array to check
        warning: If True, show warning for non-float64 arrays
        tag: Description tag for the array in warning message
    """
    if arr.dtype != np.float64 and warning:
        warnings.warn(f"{tag} is not np.float64 but {arr.dtype}", UserWarning)


def warn_outside_wavenumber_grid(
    nu_lines: Union[np.ndarray, jnp.ndarray], nu_grid: Union[np.ndarray, jnp.ndarray]
) -> None:
    """Check if all line centers are within the wavenumber grid.

    Args:
        nu_lines: Line center wavenumbers in cm⁻¹
        nu_grid: Wavenumber grid in cm⁻¹

    Note:
        For MODIT/DIT, lines outside the wavenumber grid contribute to the
        grid edges, which can cause artifacts. This warning often occurs
        when using non-negative margin values in molecular databases.
        See Issue #190: https://github.com/HajimeKawahara/exojax/issues/190
    """
    if np.min(nu_lines) < np.min(nu_grid) or np.max(nu_lines) > np.max(nu_grid):
        warnings.warn("Some of the line centers are outside of the wavenumber grid.")
        warnings.warn(
            "All of the line center should be within wavenumber grid for PreMODIT/MODIT/DIT."
        )
