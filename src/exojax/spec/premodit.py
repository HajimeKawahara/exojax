"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from exojax.spec.lsd import npgetix, npgetix_exp, npadd3D_multi_index
from exojax.utils.constants import hcperk, Tref
from exojax.spec.modit_scanfft import calc_xsection_from_lsd_scanfft
from exojax.spec.set_ditgrid import ditgrid_log_interval, ditgrid_linear_interval
from exojax.utils.constants import Tref
from exojax.utils.indexing import uniqidx_neibouring
from exojax.spec import normalized_doppler_sigma


@jit
def xsvector(T, P, nsigmaD, lbd, R, pmarray, nu_grid, elower_grid,
             multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt):
    """compute cross section vector, with scan+fft

    Args:
        T (_type_): temperature in Kelvin
        P (_type_): pressure in bar
        nsigmaD: normalized doplar STD
        lbd (_type_): log biased line shape density (LBD)
        R (_type_): spectral resolution
        nu_grid (_type_): wavenumber grid
        elower_grid (_type_): E lower grid
        multi_index_uniqgrid (_type_): multi index of unique broadening parameter grid
        ngamma_ref_grid (_type_): normalized pressure broadening half-width
        n_Texp_grid (_type_): temperature exponent grid
        qt (_type_): partirion function ratio

    Returns:
        jnp.array: cross section in cgs vector
    """
    Slsd = unbiased_lsd(lbd, T, nu_grid, elower_grid, qt)
    ngamma_grid = unbiased_ngamma_grid(T, P, ngamma_ref_grid, n_Texp_grid,
                                       multi_index_uniqgrid)
    log_ngammaL_grid = jnp.log(ngamma_grid)
    xs = calc_xsection_from_lsd_scanfft(Slsd, R, pmarray, nsigmaD, nu_grid,
                                log_ngammaL_grid)
    return xs


@jit
def xsmatrix(Tarr, Parr, R, pmarray, lbd, nu_grid, ngamma_ref_grid,
             n_Texp_grid, multi_index_uniqgrid, elower_grid, Mmol, qtarr):
    """compute cross section matrix given atmospheric layers, with scan+fft

    Args:
        Tarr (_type_): temperature layers
        Parr (_type_): pressure layers
        R (float): spectral resolution
        pmarray (_type_): pmarray
        lbd (_type_): log biased line shape density
        nu_grid (_type_): wavenumber grid
        ngamma_ref_grid (_type_): normalized half-width grid
        n_Texp_grid (_type_): temperature exponent grid
        multi_index_uniqgrid (_type_): multi index for uniq broadpar grid
        elower_grid (_type_): Elower grid
        Mmol (_type_): molecular mass
        qtarr (_type_): partition function ratio layers

    Returns:
        jnp.array : cross section matrix (Nlayer, N_wavenumber)
    """
    nsigmaD = vmap(normalized_doppler_sigma, (0, None, None), 0)(Tarr, Mmol, R)
    Slsd = vmap(unbiased_lsd, (None, 0, None, None, 0), 0)(lbd, Tarr, nu_grid,
                                                           elower_grid, qtarr)
    ngamma_grid = vmap(unbiased_ngamma_grid, (0, 0, None, None, None),
                       0)(Tarr, Parr, ngamma_ref_grid, n_Texp_grid,
                          multi_index_uniqgrid)
    log_ngammaL_grid = jnp.log(ngamma_grid)
    xsm = vmap(calc_xsection_from_lsd_scanfft, (0, None, None, 0, None, 0),
               0)(Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid)
    return xsm


def parallel_merge_grids(grid1, grid2):
    """Merge two different grids into one grid in parallel, in a C-contiguous RAM mapping.
    
    Args:
        grid1: grid 1
        grid2: grid 2
        
    Returns:
        merged grid (len(grid1),2)
            
    """
    if len(grid1) != len(grid2):
        raise ValueError("lengths for grid1 and grid2 are different.")

    merged_grid = np.ascontiguousarray(np.vstack([grid1, grid2]).T)
    return merged_grid


def make_broadpar_grid(ngamma_ref,
                       n_Texp,
                       Ttyp,
                       dit_grid_resolution=0.2,
                       adopt=True):
    """ make grids of normalized half-width at reference and temperature exoponent

    Args:
        ngamma_ref (nd array): n_Texp: temperature exponent (n_Texp, n_air, etc)
        n_Texp (nd array): temperature exponent (n_Texp, n_air, etc)
        Ttyp: typical or maximum temperature
        dit_grid_resolution (float, optional): DIT grid resolution. Defaults to 0.2.
        adopt (bool, optional): if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be dit_grid_resolution exactly.  Defaults to True.
        
    Returns:
        nd array: ngamma_ref_grid, grid of normalized half-width at reference 
        nd array: n_Texp_grid, grid of temperature exponent (n_Texp, n_air, etc)
        
    """
    ngamma_ref_grid = ditgrid_log_interval(
        ngamma_ref, dit_grid_resolution=dit_grid_resolution, adopt=adopt)
    weight = np.log(Ttyp) - np.log(Tref)
    n_Texp_grid = ditgrid_linear_interval(
        n_Texp,
        dit_grid_resolution=dit_grid_resolution,
        weight=weight,
        adopt=adopt)
    return ngamma_ref_grid, n_Texp_grid


def broadpar_getix(ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid):
    """get indices and contribution of broadpar
    
    Args:
        ngamma_ref: normalized half-width at reference 
        ngamma_ref_grid: grid of normalized half-width at reference 
        n_Texp: temperature exponent (n_Texp, n_air, etc)
        n_Texp_grid: grid of temperature exponent (n_Texp, n_air, etc)
        
    Returns:
        multi_index_lines
        multi_cont_lines
        uidx_lines
        neighbor_uidx
        multi_index_uniqgrid 
        number of broadpar gird
        
    Examples:
        
        >>> multi_index_lines, multi_cont_lines, uidx_lines, neighbor_uidx, multi_index_uniqgrid, Ng = broadpar_getix(
        >>> ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)
        >>> iline_interest = 1 # we wanna investiget this line
        >>> print(multi_index_lines[iline_interest]) # multi index from a line
        >>> print(multi_cont_lines[iline_interest]) # multi contribution from a line 
        >>> print(uidx_lines[iline_interest]) # uniq index of a line
        >>> print(multi_index_uniqgrid[uniq_index]) # multi index from uniq_index
        >>> ui,uj,uk=neighbor_uidx[uniq_index, :] # neighbour uniq index
    """
    cont_ngamma_ref, index_ngamma_ref = npgetix(ngamma_ref, ngamma_ref_grid)
    cont_n_Texp, index_n_Texp = npgetix(n_Texp, n_Texp_grid)
    multi_index_lines = parallel_merge_grids(index_ngamma_ref, index_n_Texp)
    multi_cont_lines = parallel_merge_grids(cont_ngamma_ref, cont_n_Texp)
    uidx_lines, neighbor_indices, multi_index_uniqgrid = uniqidx_neibouring(
        multi_index_lines)
    Ng_broadpar = len(multi_index_uniqgrid)
    return multi_index_lines, multi_cont_lines, uidx_lines, neighbor_indices, multi_index_uniqgrid, Ng_broadpar


def compute_dElower(T, interval_contrast=0.1):
    """ compute a grid interval of Elower given the grid interval of line strength

    Args: 
        T: temperature in Kelvin
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
        Required dElower
    """
    return interval_contrast * np.log(10.0) * T / hcperk


def make_elower_grid(Tmax, elower, interval_contrast):
    """compute E_lower grid given interval_contrast of line strength

    Args: 
        Tmax: max temperature in Kelvin
        elower: E_lower
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
        grid of E_lower given interval_contrast

    """
    dE = compute_dElower(Tmax, interval_contrast)
    min_elower = np.min(elower)
    max_elower = np.max(elower)
    Ng_elower = int((max_elower - min_elower) / dE) + 2
    return min_elower + np.arange(Ng_elower) * dE


def generate_lbd(line_strength_ref, nu_lines, nu_grid, ngamma_ref,
                 ngamma_ref_grid, n_Texp, n_Texp_grid, elower, elower_grid,
                 Ttyp):
    """generate log-biased line shape density (LBD)

    Args:
        line_strength_ref: line strength at reference temperature 296K, Sij0
        nu_lines (_type_): _description_
        nu_grid (_type_): _description_
        ngamma_ref (_type_): _description_
        ngamma_ref_grid (_type_): _description_
        n_Texp (_type_): _description_
        n_Texp_grid (_type_): _description_
        elower (_type_): _description_
        elower_grid (_type_): _description_
        Ttyp (_type_): _description_

    Returns:
        jnp array: log-biased line shape density (LBD)
        jnp.array: multi_index_uniqgrid (number of unique broadpar, 2)
        
    Examples:

        >>> lbd, multi_index_uniqgrid = generate_lbd(mdb.Sij0, mdb.nu_lines, nu_grid, ngamma_ref,
        >>>               ngamma_ref_grid, mdb.n_Texp, n_Texp_grid, mdb.elower,
        >>>               elower_grid, Ttyp)
        >>> ngamma_ref = ngamma_ref_grid[multi_index_uniqgrid[:,0]] # ngamma ref for the unique broad par
        >>> n_Texp = n_Texp_grid[multi_index_uniqgrid[:,0]] # temperature exponent for the unique broad par
        
    """
    logmin = -np.inf
    cont_nu, index_nu = npgetix(nu_lines, nu_grid)
    cont_elower, index_elower = npgetix_exp(elower, elower_grid, Ttyp)
    multi_index_lines, multi_cont_lines, uidx_bp, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)

    Ng_nu = len(nu_grid)

    # We extend the LBD grid to +1 along elower direction. See #273
    Ng_elower_plus_one = len(elower_grid) + 1

    lbd = np.zeros((Ng_nu, Ng_broadpar, Ng_elower_plus_one), dtype=np.float64)
    lbd = npadd3D_multi_index(lbd, line_strength_ref, cont_nu, index_nu,
                              cont_elower, index_elower, uidx_bp,
                              multi_cont_lines, neighbor_uidx)
    lbd[lbd > 0.0] = np.log(lbd[lbd > 0.0])
    lbd[lbd == 0.0] = logmin

    # Removing the extended grid of elower. See #273
    lbd = lbd[:, :, 0:-1]

    return jnp.array(lbd), multi_index_uniqgrid


def logf_bias(elower_in, T):
    """logarithm f bias function
    
    Args:
        elower_in: Elower in cm-1
        T: temperature for unbiasing in Kelvin

    Returns:
        logarithm of bias f function

    """
    return -hcperk * elower_in * (1. / T - 1. / Tref)


def g_bias(nu_in, T):
    """
    Args:
        nu_in: wavenumber in cm-1
        T: txbemperature for unbiasing in Kelvin

    Returns:
        bias g function

    """
    #return jnp.expm1(-hcperk*nu_in/T) / jnp.expm1(-hcperk*nu_in/Tref)
    return (1.0 - jnp.exp(-hcperk * nu_in / T)) / (
        1.0 - jnp.exp(-hcperk * nu_in / Tref))


def unbiased_lsd(lbd, T, nu_grid, elower_grid, qt):
    """ unbias the biased LSD

    Args:
        lbd_biased: log biased LSD
        T: temperature for unbiasing in Kelvin
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qt: partition function ratio Q(T)/Q(Tref)

    Returns:
        LSD, shape = (number_of_wavenumber_bin, number_of_broadening_parameters)
        
    """
    Slsd = jnp.sum(jnp.exp(logf_bias(elower_grid, T) + lbd), axis=-1)
    return (Slsd.T * g_bias(nu_grid, T) / qt).T


def unbiased_ngamma_grid(T, P, ngamma_ref_grid, n_Texp_grid,
                         multi_index_uniqgrid):
    """compute unbiased ngamma grid

    Args:
        T: temperature in Kelvin
        P: pressure in bar
        ngamma_ref_grid : pressure broadening half width at reference 
        n_Texp_grid : temperature exponent at reference
        multi_index_uniqgrid: multi index of unique broadening parameter

    Returns:
        pressure broadening half width at temperature T and pressure P 
    """
    ngamma_ref_g = ngamma_ref_grid[multi_index_uniqgrid[:, 0]]
    n_Texp_g = n_Texp_grid[multi_index_uniqgrid[:, 1]]
    return ngamma_ref_g * (T / Tref)**(-n_Texp_g) * P


# def lowpass(fftval, compress_rate):
#     """lowpass filter for the biased LSD

#     """
#     Nnu, Nbatch = np.shape(fftval)
#     Ncut = int(float(Nnu) / float(compress_rate))
#     lowpassed_fftval = fftval[:Ncut, :]
#     high_freq_norm_squared = np.sum(np.abs(fftval[Ncut:, :])**2, axis=0)
#     lowpassed_fftval[0, :] = np.sqrt(
#         np.abs(lowpassed_fftval[0, :])**2 + high_freq_norm_squared)
#     return lowpassed_fftval

# def unbiased_lsd_lowpass(FT_Slsd_biased, T, nu_grid, elower_grid, qr):
#     """ unbias the biased LSD lowpass filtered

#     Args:

#     Returns:
#         LSD (unbiased)

#     """
#     Nnu = int(len(nu_grid) / 2)
#     eunbias_FTSlsd = jnp.sum(f_bias(elower_grid, T) * FT_Slsd_biased, axis=1)
#     Nft = len(eunbias_FTSlsd)
#     eunbias_FTSbuf = jnp.hstack([eunbias_FTSlsd, jnp.zeros(Nnu - Nft + 1)])
#     eunbias_Slsd = jnp.fft.irfft(eunbias_FTSbuf)
#     return g_bias(nu_grid, T) * eunbias_Slsd / qr(T)
