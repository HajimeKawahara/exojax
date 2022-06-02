"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from exojax.spec.lsd import npgetix, npgetix_exp
from exojax.utils.constants import hcperk, Tref
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.modit import calc_xsection_from_lsd
from exojax.spec.exomol import gamma_exomol
from exojax.spec.set_ditgrid import ditgrid_log_interval, ditgrid_linear_interval
from exojax.utils.constants import Tref
from exojax.utils.indexing import uniqidx_neibouring

def diad_merge_grids(grid1, grid2):
    """merge two different grids into one grid using diad
    
    Args:
        grid1: grid 1
        grid2: grid 2
        
    Returns:
        merged grid (len(grid1)*len(grid2),2)
        
    Examples:
        >>>  broadpar_grid = merge_grids(ngamma_ref_grid,n_Texp_grid)
            
    """
    X, Y = np.meshgrid(grid1, grid2)
    Ng = np.shape(X)[0] * np.shape(X)[1]
    merged_grid = (np.array([X, Y]).T).reshape(Ng, 2)
    return merged_grid


def parallel_merge_grids(grid1, grid2):
    """merge two different grids into one grid in parallel
    
    Args:
        grid1: grid 1
        grid2: grid 2
        
    Returns:
        merged grid (len(grid1),2)
            
    """
    if len(grid1) != len(grid2):
        assert ValueError("lengths for grid1 and grid2 are different.")

    merged_grid = np.vstack([grid1, grid2]).T
    return merged_grid


def broadpar_getix(ngamma_ref,
                   n_Texp,
                   Ttyp,
                   dit_grid_resolution=0.2,
                   adopt=True):
    """get indices and contribution of broadpar
    
    Args:
        ngamma_ref: normalized half-width at reference 
        n_Texp: temperature exponent (n_Texp, n_air, etc)
        Ttyp: typical or maximum temperature
        dit_grid_resolution: DIT grid resolution
        adopt: if True, min, max grid points are used at min and max values of x.
        In this case, the grid width does not need to be dit_grid_resolution exactly.
        
    Returns:
        
    
    """

    ngamma_ref_grid = ditgrid_log_interval(
        ngamma_ref, dit_grid_resolution=dit_grid_resolution, adopt=adopt)
    weight = np.log(Ttyp) - np.log(Tref)
    n_Texp_grid = ditgrid_linear_interval(
        n_Texp,
        dit_grid_resolution=dit_grid_resolution,
        weight=weight,
        adopt=adopt)

    cont_ngamma_ref, index_ngamma_ref = npgetix(ngamma_ref,
                                                ngamma_ref_grid)  # Nline
    cont_n_Texp, index_n_Texp = npgetix(n_Texp, n_Texp_grid)  # Nline

    multi_index_lines = parallel_merge_grids(
        index_ngamma_ref, index_n_Texp)  #(Nline, 2) but index
    multi_cont_lines = parallel_merge_grids(cont_ngamma_ref, cont_n_Texp)
    uidx_lines, neighbor_indices, multi_index_uniqgrid = uniqidx_neibouring(
        multi_index_lines)

    return multi_index_lines, multi_cont_lines, uidx_lines, neighbor_indices, multi_index_uniqgrid


def compute_dElower(T, interval_contrast=0.1):
    """ compute a grid interval of Elower given the grid interval of line strength

    Args: 
        T: temperature in Kelvin
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
        Required dEloweredt mru 
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


def make_lbd(line_strength_ref, cont_nu, index_nu, Ng_nu, elower, elower_grid,
             Ttyp):
    """make logarithm biased LSD (LBD) array 

    Args:
        line_strength_ref: line strength at the refrence temepreature Tref (should be F64)
        cont_nu: (wavenumber contribution) jnp.array
        index_nu: (wavenumber index) jnp.array
        Ng_nu: number of wavenumber bins (len(nu_grid))
        elower: E lower
        elower_grid: E lower grid
        
        uidx_broadpar: broadening parameter index
        Ttyp: typical temperature you will use.

    Returns:
        lbd

    Notes: 
        LBD (jnp array), contribution fow wavenumber, index for wavenumber

    """
    logmin = -np.inf
    lsd = np.zeros((Ng_nu, np.max(uidx_broadpar) + 1, len(elower_grid)),
                   dtype=np.float64)
    cont_elower, index_elower = npgetix_exp(elower, elower_grid, Ttyp)
    lsd = npadd3D(lsd, Sij0, cont_nu, index_nu, cont_broadpar, index_broadpar,
                  cont_elower, index_elower)
    lsd[lsd > 0.0] = np.log(lsd[lsd > 0.0])
    lsd[lsd == 0.0] = logmin
    return jnp.array(lsd)


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
        T: temperature for unbiasing in Kelvin

    Returns:
        bias g function

    """
    #return jnp.expm1(-hcperk*nu_in/T) / jnp.expm1(-hcperk*nu_in/Tref)
    return (1.0 - jnp.exp(-hcperk * nu_in / T)) / (
        1.0 - jnp.exp(-hcperk * nu_in / Tref))


def unbiased_lsd(lbd_biased, T, nu_grid, elower_grid, qr):
    """ unbias the biased LSD

    Args:
        lbd_biased: log biased LSD
        T: temperature for unbiasing in Kelvin
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qr: partition function ratio Q(T)/Q(Tref)

    Returns:
        Unbiased 2D LSD, shape = (number_of_wavenumber_bin, number_of_broadening_parameters)

    """
    Nnu = int(len(nu_grid) / 2)
    eunbias_lbd = jnp.sum(jnp.exp(logf_bias(elower_grid, T) + lbd_biased),
                          axis=-1)
    return (eunbias_lbd.T * g_bias(nu_grid, T) / qr(T)).T


def lowpass(fftval, compress_rate):
    """lowpass filter for the biased LSD

    """
    Nnu, Nbatch = np.shape(fftval)
    Ncut = int(float(Nnu) / float(compress_rate))
    lowpassed_fftval = fftval[:Ncut, :]
    high_freq_norm_squared = np.sum(np.abs(fftval[Ncut:, :])**2, axis=0)
    lowpassed_fftval[0, :] = np.sqrt(
        np.abs(lowpassed_fftval[0, :])**2 + high_freq_norm_squared)
    return lowpassed_fftval


def unbiased_lsd_lowpass(FT_Slsd_biased, T, nu_grid, elower_grid, qr):
    """ unbias the biased LSD lowpass filtered

    Args:

    Returns:
        LSD (unbiased)

    """
    Nnu = int(len(nu_grid) / 2)
    eunbias_FTSlsd = jnp.sum(f_bias(elower_grid, T) * FT_Slsd_biased, axis=1)
    Nft = len(eunbias_FTSlsd)
    eunbias_FTSbuf = jnp.hstack([eunbias_FTSlsd, jnp.zeros(Nnu - Nft + 1)])
    eunbias_Slsd = jnp.fft.irfft(eunbias_FTSbuf)
    return g_bias(nu_grid, T) * eunbias_Slsd / qr(T)


######################################

if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    import matplotlib.pyplot as plt
    from exojax.spec.lsd import npadd1D, npgetix
    from exojax.spec.hitran import SijT
    from exojax.spec.initspec import init_premodit
    from exojax.spec.hitran import normalized_doppler_sigma
    from exojax.spec import molinfo

    nus = np.logspace(np.log10(6020.0),
                      np.log10(6080.0),
                      40000,
                      dtype=np.float64)
    mdb = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/',
                          nus,
                          gpu_transfer=False)

    Ttyp = 1500.0
    broadpar = np.array([mdb._n_Texp, mdb._alpha_ref]).T
    cont_nu, index_nu, elower_grid, uidx_broadpar, uniq_broadpar, R, pmarray = init_premodit(
        mdb.nu_lines, nus, mdb._elower, Ttyp=Ttyp, broadpar=broadpar)

    lbd = make_lbd3D_uniqidx(mdb.Sij0, cont_nu, index_nu, len(nus),
                             mdb._elower, elower_grid, uidx_broadpar, Ttyp)

    Tfix = 1000.0
    Pfix = 1.e-3
    molmassCH4 = molinfo.molmass("CH4")

    nsigmaD = normalized_doppler_sigma(Tfix, molmassCH4, R)
    qt = mdb.qr_interp(Tfix)
    print(uniq_broadpar[:, 0], uniq_broadpar[:, 1], "ui")
    gammaL = gamma_exomol(Pfix, Tfix, uniq_broadpar[:, 0], uniq_broadpar[:, 1])
    print(gammaL, "gma")
#    log_ngammaL_grid = jnp.log(gammaL/(/R))
#    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
#    Slsd=unbiased_lsd(lbd,Tfix,nus,elower_grid,qt)
#    xs = calc_xsection_from_lsd(Slsd, R, pmarray, nsigmaD, nus, log_ngammaL_grid)
