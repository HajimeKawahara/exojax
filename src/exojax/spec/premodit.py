"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from exojax.spec.lsd import npgetix, npgetix_exp, npadd3D_multi_index
from exojax.utils.constants import hcperk, Tref_original
from exojax.spec.modit_scanfft import calc_xsection_from_lsd_scanfft
from exojax.spec.set_ditgrid import ditgrid_log_interval, ditgrid_linear_interval
from exojax.utils.constants import Tref_original
from exojax.utils.indexing import uniqidx_neibouring
from exojax.spec import normalized_doppler_sigma
from exojax.spec.lbd import lbd_coefficients

@jit
def xsvector(T, P, nsigmaD, lbd_zeroth, lbd_first, Twt, R, pmarray, nu_grid, elower_grid,
             multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt):
    """compute cross section vector, with scan+fft, using the first Taylor expansion

    Args:
        T (_type_): temperature in Kelvin
        P (_type_): pressure in bar
        nsigmaD: normalized doplar STD
        lbd_zeroth (_type_): log biased line shape density (LBD), zeroth coefficient
        lbd_first (_type_): log biased line shape density (LBD), first coefficient
        Twt: temperature used in the weight point
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
    Slsd = unbiased_lsd_first(lbd_zeroth, lbd_first, T, Twt, nu_grid, elower_grid, qt)
    ngamma_grid = unbiased_ngamma_grid(T, P, ngamma_ref_grid, n_Texp_grid,
                                       multi_index_uniqgrid)
    log_ngammaL_grid = jnp.log(ngamma_grid)
    xs = calc_xsection_from_lsd_scanfft(Slsd, R, pmarray, nsigmaD, nu_grid,
                                        log_ngammaL_grid)
    return xs

@jit
def xsvector_zeroth(T, P, nsigmaD, lbd_zeroth, R, pmarray, nu_grid, elower_grid,
             multi_index_uniqgrid, ngamma_ref_grid, n_Texp_grid, qt):
    """compute cross section vector, with scan+fft, using the zero-th Taylor expansion 

    Args:
        T (_type_): temperature in Kelvin
        P (_type_): pressure in bar
        nsigmaD: normalized doplar STD
        lbd_zeroth (_type_): log biased line shape density (LBD)
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
    Slsd = unbiased_lsd_zeroth(lbd_zeroth, T, nu_grid, elower_grid, qt)
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
    Slsd = vmap(unbiased_lsd_zeroth, (None, 0, None, None, 0),
                0)(lbd, Tarr, nu_grid, elower_grid, qtarr)
    ngamma_grid = vmap(unbiased_ngamma_grid, (0, 0, None, None, None),
                       0)(Tarr, Parr, ngamma_ref_grid, n_Texp_grid,
                          multi_index_uniqgrid)
    log_ngammaL_grid = jnp.log(ngamma_grid)
    xsm = vmap(calc_xsection_from_lsd_scanfft, (0, None, None, 0, None, 0),
               0)(Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid)
    return xsm


def parallel_merge_grids(grid1, grid2):
    """merge two different grids into one grid in parallel
    
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
    weight = np.log(Ttyp) - np.log(Tref_original)
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


def generate_lbd(line_strength_ref,
                 nu_lines,
                 nu_grid,
                 ngamma_ref,
                 ngamma_ref_grid,
                 n_Texp,
                 n_Texp_grid,
                 elower,
                 elower_grid,
                 Twt,
                 diffmode=0):
    """generate log-biased line shape density (LBD)

    Args:
        line_strength_ref (_type_): _description_
        nu_lines (_type_): _description_
        nu_grid (_type_): _description_
        ngamma_ref (_type_): _description_
        ngamma_ref_grid (_type_): _description_
        n_Texp (_type_): _description_
        n_Texp_grid (_type_): _description_
        elower (_type_): _description_
        elower_grid (_type_): _description_
        Twt: temperature used for the weight coefficient computation 
        diffmode (int): i-th Taylor expansion is used for the weight, default is 1.

    Returns:
        jnp array: the zeroth coeff of log-biased line shape density (LBD)
        jnp array: the first coeff of log-biased line shape density (LBD), None for diffmode=0
        jnp.array: multi_index_uniqgrid (number of unique broadpar, 2)
        
    Examples:

        >>> lbd_zeroth, lbd_first, multi_index_uniqgrid = generate_lbd(mdb.Sij0, mdb.nu_lines, nu_grid, ngamma_ref,
        >>>               ngamma_ref_grid, mdb.n_Texp, n_Texp_grid, mdb.elower,
        >>>               elower_grid, Twp)
        >>> ngamma_ref = ngamma_ref_grid[multi_index_uniqgrid[:,0]] # ngamma ref for the unique broad par
        >>> n_Texp = n_Texp_grid[multi_index_uniqgrid[:,0]] # temperature exponent for the unique broad par
        
    """
    cont_nu, index_nu = npgetix(nu_lines, nu_grid)
    multi_index_lines, multi_cont_lines, uidx_bp, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)
    Ng_nu = len(nu_grid)

    # We extend the LBD grid to +1 along elower direction. See Issue #273
    Ng_elower_plus_one = len(elower_grid) + 1
    lbd_zeroth = np.zeros((Ng_nu, Ng_broadpar, Ng_elower_plus_one),
                          dtype=np.float64)
    if diffmode == 0:
        zeroth_coeff_elower, index_elower = npgetix_exp(
            elower, elower_grid, Twt)
        lbd_first = None
    elif diffmode == 1:
        zeroth_coeff_elower, first_coeff_elower, index_elower = lbd_coefficients(
            elower, elower_grid, Tref_original, Twt)
        lbd_first = np.zeros((Ng_nu, Ng_broadpar, Ng_elower_plus_one),
                             dtype=np.float64)
        lbd_first = npadd3D_multi_index(lbd_first,
                                        line_strength_ref,
                                        cont_nu,
                                        index_nu,
                                        first_coeff_elower,
                                        index_elower,
                                        uidx_bp,
                                        multi_cont_lines,
                                        neighbor_uidx,
                                        sumz=0.0)
        #lbd_first = convert_to_jnplog(lbd_first)
        lbd_first = jnp.array(lbd_first[:, :, 0:-1])
    else:
        raise ValueError("diffmode = 0 or 1 is allowed.", UserWarning)

    lbd_zeroth = npadd3D_multi_index(lbd_zeroth, line_strength_ref, cont_nu,
                                     index_nu, zeroth_coeff_elower,
                                     index_elower, uidx_bp, multi_cont_lines,
                                     neighbor_uidx, sumz=1.0)

    lbd_zeroth = convert_to_jnplog(lbd_zeroth)
    return lbd_zeroth, lbd_first, multi_index_uniqgrid


def convert_to_jnplog(lbd_nth):
    """compute log and convert to jnp

    Notes:
        This function is used to avoid an overflow when using FP32 in JAX

    Args:
        lbd_nth (ndarray): n-th coefficient (non-log) LBD

    Returns:
        jnp.array: log form of n-th coefficient LBD
    """
    logmin = -np.inf
    lbd_nth[lbd_nth > 0.0] = np.log(lbd_nth[lbd_nth > 0.0])
    lbd_nth[lbd_nth == 0.0] = logmin
    # Removing the extended grid of elower. See Issue #273
    lbd_nth = jnp.array(lbd_nth[:, :, 0:-1])
    return lbd_nth


def logf_bias(elower_in, T):
    """logarithm f bias function
    
    Args:
        elower_in: Elower in cm-1
        T: temperature for unbiasing in Kelvin

    Returns:
        logarithm of bias f function

    """
    return -hcperk * elower_in * (1. / T - 1. / Tref_original)


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
        1.0 - jnp.exp(-hcperk * nu_in / Tref_original))


def unbiased_lsd_zeroth(lbd_zeroth, T, nu_grid, elower_grid, qt):
    """ unbias the biased LSD

    Args:
        lbd_zeroth: the zeroth coeff of log-biased line shape density (LBD)
        T: temperature for unbiasing in Kelvin
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qt: partition function ratio Q(T)/Q(Tref)

    Returns:
        LSD (0th), shape = (number_of_wavenumber_bin, number_of_broadening_parameters)
        
    """
    Slsd = jnp.sum(jnp.exp(logf_bias(elower_grid, T) + lbd_zeroth), axis=-1)
    return (Slsd.T * g_bias(nu_grid, T) / qt).T


def unbiased_lsd_first(lbd_zeroth, lbd_first, T, Twt, nu_grid, elower_grid, qt):
    """ unbias the biased LSD

    Args:
        lbd_zeroth: the zeroth coeff of log-biased line shape density (LBD)
        lbd_first: the first coeff of log-biased line shape density (LBD)
        T: temperature for unbiasing in Kelvin
        Twt: Temperature at the weight point
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qt: partition function ratio Q(T)/Q(Tref)

    Returns:
        LSD, shape = (number_of_wavenumber_bin, number_of_broadening_parameters)
        
    """
    lfb = logf_bias(elower_grid, T)
    Slsd_zeroth = jnp.sum(jnp.exp(lfb + lbd_zeroth), axis=-1)
    unbiased_coeff = jnp.exp(lfb)*lbd_first*(1.0 / T - 1.0 / Twt)  # f*w1
    Slsd_first = jnp.sum(unbiased_coeff,axis=-1)  # sum_l[ f*w1(t-twt) ]
    Slsd = Slsd_zeroth + Slsd_first
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
    return ngamma_ref_g * (T / Tref_original)**(-n_Texp_g) * P
