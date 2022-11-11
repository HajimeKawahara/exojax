"""Initialization for opacity computation.

The functions in this module are a wrapper for initialization processes
for opacity computation.
"""
import jax.numpy as jnp
import numpy as np
import warnings

from exojax.spec.lsd import npgetix
from exojax.spec.make_numatrix import make_numatrix0
from exojax.utils.instfunc import resolution_eslog
from exojax.spec.premodit import make_elower_grid
from exojax.spec.premodit import make_broadpar_grid
from exojax.spec.premodit import generate_lbd


def init_lpf(nu_lines, nu_grid):
    """Initialization for LPF.

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        numatrix [Nline,Nnu]
    """
    numatrix = make_numatrix0(nu_grid, nu_lines, warning=True)
    return numatrix


def init_dit(nu_lines, nu_grid, warning=False):
    """Initialization for DIT. i.e. Generate nu contribution and index for the
        line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        cont (contribution) jnp.array
        index (index) jnp.array
        pmarray: (+1.,-1.) array whose length of len(nu_grid)+1

    Note:
        cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """
    warn_dtype64(nu_lines, warning, tag='nu_lines')
    warn_dtype64(nu_grid, warning, tag='nu_grid')
    warn_outside_wavenumber_grid(nu_lines, nu_grid)

    cont, index = npgetix(nu_lines, nu_grid)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0

    return jnp.array(cont), jnp.array(index), pmarray


def init_modit(nu_lines, nu_grid, warning=False):
    """Initialization for MODIT. i.e. Generate nu contribution and index for
    the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        cont: (contribution for q) jnp.array
        index: (index for q) jnp.array
        spectral_resolution: spectral resolution (R)
        pmarray: (+1.,-1.) array whose length of len(nu_grid)+1

    Note:
        cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero. dq is computed using numpy not jnp.numpy. If you use jnp, you might observe a significant residual because of the float32 truncation error.
    """
    warn_dtype64(nu_lines, warning, tag='nu_lines')
    warn_dtype64(nu_grid, warning, tag='nu_grid')
    warn_outside_wavenumber_grid(nu_lines, nu_grid)

    spectral_resolution = resolution_eslog(nu_grid)
    cont, index = npgetix(nu_lines, nu_grid)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0

    return jnp.array(cont), jnp.array(index), spectral_resolution, jnp.array(pmarray)


def init_premodit(nu_lines,
                  nu_grid,
                  elower,
                  gamma_ref,
                  n_Texp,
                  line_strength_ref,
                  Ttyp,
                  interval_contrast=0.1,
                  dit_grid_resolution=0.2,
                  warning=False):
    """Initialization for PreMODIT. i.e. Generate nu contribution and index for
    the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        elower: elower of lines
        gamma_ref: half-width at reference (alpha_ref for ExoMol, gamma_air for HITRAN/HITEMP etc)
        n_Texp: temperature exponent (n_Texp for ExoMol, n_air for HITRAN/HITEMP)
        line_strength_ref: line strength at reference temperature 296K, Sij0
        Ttyp: typical temperature in Kelvin
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.
        dit_grid_resolution: DIT grid resolution 

    Returns:
        cont_nu: contribution for wavenumber jnp.array
        index_nu: index for wavenumber jnp.array
        elower_grid: elower grid 
        cont_broadpar: contribution for broadening parmaeters
        index_broadpar: index for broadening parmaeters
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1


    Note:
        cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero. dq is computed using numpy not jnp.numpy. If you use jnp, you might observe a significant residual because of the float32 truncation error.
    """
    warn_dtype64(nu_lines, warning, tag='nu_lines')
    warn_dtype64(nu_grid, warning, tag='nu_grid')
    warn_dtype64(elower, warning, tag='elower')
    warn_outside_wavenumber_grid(nu_lines, nu_grid)

    R = resolution_eslog(nu_grid)
    ngamma_ref = gamma_ref / nu_lines * R
    elower_grid = make_elower_grid(Ttyp,
                                   elower,
                                   interval_contrast=interval_contrast)
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(ngamma_ref,
                                                      n_Texp,
                                                      Ttyp,
                                                      dit_grid_resolution=dit_grid_resolution)
    lbd, multi_index_uniqgrid = generate_lbd(line_strength_ref, nu_lines, nu_grid, ngamma_ref, ngamma_ref_grid,
             n_Texp, n_Texp_grid, elower, elower_grid, Ttyp)
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    return lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, jnp.array(pmarray)


def init_modit_vald(nu_linesM, nus, N_usp):
    """Initialization for MODIT for asdb from VALD

    Args:
        nu_linesM: wavenumbers of lines for each species [N_species x N_line] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        N_usp: number of species

    Returns:
        contS: (contribution) jnp.array [N_species x N_line]
        indexS: (index) jnp.array [N_species x N_line]
        R: spectral resolution
    
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
    """
    contS = np.zeros_like(nu_linesM)
    indexS = np.zeros_like(contS)
    for i in range(N_usp):
        nu_lines = nu_linesM[i]
        nu_lines_nan = np.where(nu_lines == 0, np.nan, nu_lines)
        contS[i], indexnu_dammy, R, pmarray = init_modit(
            nu_lines_nan, nus)  # np.array(a), np.array(b), c, np.array(d)
        indexS[i] = np.hstack([ indexnu_dammy[np.where(~np.isnan(nu_lines_nan))], \
               (len(nus)+1) * np.ones(len(np.where(np.isnan(nu_lines_nan))[0]), dtype='int32') ])
    contS = jnp.array(contS)
    indexS = jnp.array(indexS, dtype='int32')
    return contS, indexS, R, pmarray


def warn_dtype64(arr, warning, tag=''):
    """check arr's dtype.

    Args:
        arr: input array
        warning: True/False
        tag:
    """
    if (arr.dtype != np.float64 and warning):
        warnings.warn(tag + ' is not np.float64 but ' + str(arr.dtype))


def warn_outside_wavenumber_grid(nu_lines, nu_grid):
    """Check if all the line centers are in the wavenumber grid.

    Args:
        nu_lines: line center
        nu_grid: wavenumber grid

    Note:
        For MODIT/DIT, if the lines whose center are outside of the wavenumber grid, they contribute the edges of the wavenumber grid. This function is to check it. This warning often occurs when you set non-negative value to ``margin`` in MdbExomol, MdbHit, AdbVALD, and AdbKurucz in moldb. See `#190 <https://github.com/HajimeKawahara/exojax/issues/190>`_   for the details.
    """
    if np.min(nu_lines) < np.min(nu_grid) or np.max(nu_lines) > np.max(
            nu_grid):
        warnings.warn(
            'Some of the line centers are outside of the wavenumber grid.')
        warnings.warn(
            'All of the line center should be within wavenumber grid for PreMODIT/MODIT/DIT.'
        )
