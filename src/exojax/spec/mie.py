""" Mie scattering calculation using PyMieScatt


"""

import numpy as np
import jax.numpy as jnp
from exojax.utils.interp import interp2d_bilinear


def compute_mie_coeff_lognormal_grid(
    refractive_indices, refractive_wavenm, sigmag_arr, rg_arr, npart=1.0
):
    """computes miegrid for lognomal distribution parameters rg and sigmag

    Args:
        refractive_indices (_type_): refractive indices
        refractive_wavenm (_type_):  wavenlenth in nm for refractive indices
        sigmag_arr (1d array): sigma_g array
        rg_arr (1d array): rg array
        npart (_type_, optional): number of particulates. Defaults to 1.0.

    Returns:
        _type_: miegrid (N_rg, N_sigmag, N_refractive_indices, 7), 7 is the number of the mie coefficients
    """
    from tqdm import tqdm
    from PyMieScatt import Mie_Lognormal as mief

    cm2nm = 1.0e7
    Nwav = len(refractive_indices)
    Nsigmag = len(sigmag_arr)
    Nrg = len(rg_arr)
    Nmiecoeff = 7
    miegrid = np.zeros((Nrg, Nsigmag, Nwav, Nmiecoeff), dtype=np.complex128)

    for ind_sigmag, sigmag in enumerate(tqdm(sigmag_arr)):
        for ind_rg, rg in enumerate(tqdm(np.array(rg_arr) * cm2nm)):
            for ind_m, m in enumerate(tqdm(refractive_indices)):
                coeff = mief(m, refractive_wavenm[ind_m], sigmag, rg, npart)
                miegrid[ind_rg, ind_sigmag, ind_m, :] = coeff

    return miegrid


def save_miegrid(filename, miegrid, rg_arr, sigmag_arr):
    """save miegrid file

    Args:
        filename (str): file name
        jnp Nd array: miegrid
        jnp 1d array: array for rg
        jnp 1d array: array for sigmag

    """
    np.savez(filename, miegrid, rg_arr, sigmag_arr)


def read_miegrid(filename):
    """read miegrid file

    Args:
        filename (str): file name

    Returns:
        jnp Nd array: miegrid
        jnp 1d array: array for rg
        jnp 1d array: array for sigmag
    """
    dat = np.load(filename)
    miegrid = dat["arr_0"]
    rg_arr = dat["arr_1"]
    sigmag_arr = dat["arr_2"]
    return jnp.array(miegrid), jnp.array(rg_arr), jnp.array(sigmag_arr)


def make_miegrid_lognormal(
    pdb,
    filename,
    log_sigmagmin=-1.0,
    log_sigmagmax=1.0,
    Nsigmag=10,
    log_rg_min=-7.0,
    log_rg_max=-3.0,
    Nrg=40,
):
    """ generates miegrid assuming lognormal size distribution


    Args:
        pdb (_type_): particulates database class pdb
        filename (_type_): _description_
        log_sigmagmin (float, optional): _description_. Defaults to -1.0.
        log_sigmagmax (float, optional): _description_. Defaults to 1.0.
        Nsigmag (int, optional): _description_. Defaults to 10.
        log_rg_min (float, optional): _description_. Defaults to -7.0.
        log_rg_max (float, optional): _description_. Defaults to -3.0.
        Nrg (int, optional): _description_. Defaults to 40.
    """

    sigmag_arr = np.logspace(log_sigmagmin, log_sigmagmax, Nsigmag)
    rg_arr = np.logspace(log_rg_min, log_rg_max, Nrg)  # cm

    miegrid = compute_mie_coeff_lognormal_grid(
        pdb.refraction_index,
        pdb.refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        npart=1.0,
    )
    save_miegrid(filename, miegrid, rg_arr, sigmag_arr)


def evaluate_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr):
    """evaluates the value at rg and sigmag by interpolating miegrid

    Args:
        rg (_type_): rg parameter in lognormal distribution
        sigmag (_type_): sigma_g parameter in lognormal distribution
        miegrid (5d array): Mie grid (lognormal)
        sigmag_arr (1d array): sigma_g array
        rg_arr (1d array): rg array

    Returns:
        _type_: _description_
    """
    f = interp2d_bilinear(rg, sigmag, rg_arr, sigmag_arr, miegrid)
    return f


from jax import vmap


def evaluate_miegrid_layers(rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr):
    vmapfunc = vmap(evaluate_miegrid, (0, 0, None, None, None), 0)
    return vmapfunc(rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr)


if __name__ == "__main__":
    from exojax.spec.pardb import PdbCloud
    import jax.numpy as jnp

    pdb = PdbCloud("NH3")
    filename = "miegrid_lognorm_" + pdb.condensate + ".mgd"

    miegrid, rg_arr, sigmag_arr = read_miegrid(filename)
    rg = 3.0e-5
    sigmag = 2.0
    f = evaluate_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr)

    rg_layer = jnp.array([3.0e-5, 3.0e-5])
    sigmag_layer = jnp.array([2.0, 2.0])
    f_layer = evaluate_miegrid_layers(
        rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr
    )

    print(jnp.shape(f_layer))

    print(np.shape(f))
    exit()

    # pdb = PdbCloud("NH3")
