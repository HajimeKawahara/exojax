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
                coeff = mief(
                    m, refractive_wavenm[ind_m], sigmag, 2.0 * rg, npart
                )  # geoMean is a diameter in PyMieScatt
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
    refraction_index,
    refraction_index_wavelength_nm,
    filename,
    log_sigmagmin=-1.0,
    log_sigmagmax=1.0,
    Nsigmag=10,
    log_rg_min=-7.0,
    log_rg_max=-3.0,
    Nrg=40,
    N0=1.0,
):
    """generates miegrid assuming lognormal size distribution


    Args:
        refraction_index: complex refracion (refractive) index
        refraction_index_wavelength_nm: wavelength grid in nm
        filename (str): filename
        log_sigmagmin (float, optional): log sigma_g minimum. Defaults to -1.0.
        log_sigmagmax (float, optional): log sigma_g maximum. Defaults to 1.0.
        Nsigmag (int, optional): the number of the sigmag grid. Defaults to 10.
        log_rg_min (float, optional): log r_g (cm) minimum . Defaults to -7.0.
        log_rg_max (float, optional): log r_g (cm) minimum. Defaults to -3.0.
        Nrg (int, optional): the number of the rg grid. Defaults to 40.
        N0 (float): reference number density of the lognormal distribution, default 1.0 (cm-3)


    Note:
        n0(r) dr (cm-3) = N0/sqrt(2 p d log sigma_g)  e^{-(ln d - ln d_g)^2/(2 log^2 sigma_g)}
        d_g = 2 r_g
        d = 2 r

    """

    sigmag_arr = np.logspace(log_sigmagmin, log_sigmagmax, Nsigmag)
    rg_arr = np.logspace(log_rg_min, log_rg_max, Nrg)  # cm

    miegrid = compute_mie_coeff_lognormal_grid(
        refraction_index,
        refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        npart=N0,
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

    Note:
        beta derived here is in the unit of 1/Mm (Mega meter) for diameter
        multiply 2.e-8 to convert to 1/cm for radius.


    Returns:
        _type_: evaluated values of miegrid, output of MieQ_lognormal Bext (1/Mm), Bsca, Babs, G, Bpr, Bback, Bratio (wavenumber, number of mieparams)
    """
    mieparams = interp2d_bilinear(rg, sigmag, rg_arr, sigmag_arr, miegrid)
    return mieparams


from jax import vmap


def evaluate_miegrid_layers(rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr):
    """_summary_

    Args:
        rg_layer (1d array): layer wise rg parameters
        sigmag_layer (1d array): layer wise sigmag parameters
        miegrid (5d array): Mie grid (lognormal)
        rg_arr (1d array): rg array
        sigmag_arr (1d array): sigma_g array

    Returns:
        _type_: evaluated values of miegrid, output of MieQ_lognormal Bext (1/Mm) Bsca, Babs, G, Bpr, Bback, Bratio (wavenumber, number of mieparams)
    """

    vmapfunc = vmap(evaluate_miegrid, (0, 0, None, None, None), 0)
    return vmapfunc(rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr)


def compute_mieparams(rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr, N0):
    """computes Mie parameters i.e. extinction coeff, sinigle scattering albedo, asymmetric factor

    Args:
        rg_layer (1d array): layer wise rg parameters
        sigmag_layer (1d array): layer wise sigmag parameters
        miegrid (5d array): Mie grid (lognormal)
        rg_arr (1d array): rg array
        sigmag_arr (1d array): sigma_g array
        N0: reference number density of the condensates cm-3

    Note:
        Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction

    Returns:
        beta0_extinction, volume extinction coefficient (1/cm) normalized by the reference numbver density N0
        omega0, single scattering albedo
        g, asymmetric factor (mean g)
    """

    mieparams = evaluate_miegrid_layers(
        rg_layer, sigmag_layer, miegrid, rg_arr, sigmag_arr
    )
    convfactor = 2.0e-8 / N0  # conversiont to cgs
    beta0_extinction = convfactor * mieparams[:, :, 0]  # (layer, wav)
    omega0 = mieparams[:, :, 1] / mieparams[:, :, 0]
    g = mieparams[:, :, 3]

    return beta0_extinction, omega0, g


if __name__ == "__main__":
    from exojax.spec.pardb import PdbCloud
    import jax.numpy as jnp

    pdb = PdbCloud("NH3")
    filename = ".database/particulates/virga/miegrid_lognorm_" + pdb.condensate + ".mg"
    make_miegrid_lognormal(
        pdb.refraction_index, pdb.refraction_index_wavelength_nm, filename
    )

    # exit()
    # load
    miegrid, rg_arr, sigmag_arr = read_miegrid(filename + ".npz")
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
