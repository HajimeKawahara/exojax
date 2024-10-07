""" Mie scattering calculation using PyMieScatt


"""

import numpy as np
import jax.numpy as jnp
from exojax.utils.interp import interp2d_bilinear
from exojax.special.lognormal import cubeweighted_pdf
from exojax.special.lognormal import cubeweighted_mean
from exojax.special.lognormal import cubeweighted_std
from scipy.integrate import trapezoid
import warnings


def compute_mie_coeff_lognormal_grid(
    refractive_indices, refractive_wavenm, sigmag_arr, rg_arr, N0=1.0
):
    """computes miegrid for lognomal distribution parameters rg and sigmag

    Args:
        refractive_indices (_type_): refractive indices (m = n + ik)
        refractive_wavenm (_type_):  wavenlenth in nm for refractive indices
        sigmag_arr (1d array): sigmag ($\sigma_g$) array
        rg_arr (1d array): rg ($r_g$) array
        N0 (_type_, optional): the normalization of the lognormal distribution ($N_0$). Defaults to 1.0.

    Note:
        N0 ($N_0$) is defined as the normalization of the cloud distribution,
        $n(r) pr = N_0 p(r) pr
        where $r$ is the particulate radius and
        $p(r)  = \frac{1}{sqrt{2 \pi}} \frac{1}{r \log{\sigma_g}} \exp{-\frac{(\log{r} - \log{r_g})^2}{2 \log^2{\sigma_g}}}$$
        is the lognormal distribution.
        See also https://pymiescatt.readthedocs.io/en/latest/forward.html
        they use the diameter instead (but p(d) )
        $p(d) pd = \frac{N_0}{sqrt{2 \pi}} \frac{1}{d \log{\sigma_g}} \exp{-\frac{(\log{d} - \log{d_g})^2}{2 \log^2{\sigma_g}}}$ pd
        where  and $d = 2 r$ and $d_g = 2 r_g$.
        In ExoJAX Ackerman and Marley cloud model, $N_0$ can be evaluated by `atm.amclouds.normalization_lognormal`.

    Returns:
        _type_: miegrid (N_rg, N_sigmag, N_refractive_indices, 7), 7 is the number of the mie coefficients

    """
    from tqdm import tqdm

    # from PyMieScatt import Mie_Lognormal as mief
    cm2nm = 1.0e7
    Nwav = len(refractive_indices)
    Nsigmag = len(sigmag_arr)
    Nrg = len(rg_arr)
    Nmiecoeff = 7
    miegrid = np.zeros((Nrg, Nsigmag, Nwav, Nmiecoeff))

    for ind_sigmag, sigmag in enumerate(tqdm(sigmag_arr)):
        for ind_rg, rg_nm in enumerate(tqdm(np.array(rg_arr) * cm2nm)):
            for ind_m, m in enumerate(refractive_indices):
                rgrid = auto_rgrid(rg_nm, sigmag)
                coeff = mie_lognormal_pymiescatt(
                    m, refractive_wavenm[ind_m], sigmag, rg_nm, N0, rgrid
                )
                # coeff = mief(
                #    m, refractive_wavenm[ind_m], sigmag, 2.0 * rg_nm, N0
                # )  # geoMean is a diameter (nm) in PyMieScatt
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
    sigmagmin=1.0001,
    sigmagmax=4.0,
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
        sigmagmin (float, optional): sigma_g minimum. Defaults to 1.0001.
        sigmagmax (float, optional): sigma_g maximum. Defaults to 4.0.
        Nsigmag (int, optional): the number of the sigmag grid. Defaults to 10.
        log_rg_min (float, optional): log r_g (cm) minimum . Defaults to -7.0.
        log_rg_max (float, optional): log r_g (cm) minimum. Defaults to -3.0.
        Nrg (int, optional): the number of the rg grid. Defaults to 40.
        N0 (float, optional): the normalization of the lognormal distribution ($N_0$). Defaults to 1.0.

    Note:
        N0 ($N_0$) is defined as the normalization of the cloud distribution,
        $n(r) pr = N_0 p(r) pr
        where $r$ is the particulate radius and
        $p(r)  = \frac{1}{sqrt{2 \pi}} \frac{1}{r \log{\sigma_g}} \exp{-\frac{(\log{r} - \log{r_g})^2}{2 \log^2{\sigma_g}}}$$
        is the lognormal distribution.
        See also https://pymiescatt.readthedocs.io/en/latest/forward.html
        they use the diameter instead (but p(d) )
        $p(d) pd = \frac{N_0}{sqrt{2 \pi}} \frac{1}{d \log{\sigma_g}} \exp{-\frac{(\log{d} - \log{d_g})^2}{2 \log^2{\sigma_g}}}$ pd
        where  and $d = 2 r$ and $d_g = 2 r_g$.
        In ExoJAX Ackerman and Marley cloud model, $N_0$ can be evaluated by `atm.amclouds.normalization_lognormal`.

    """
    deltasmin = np.log10(sigmagmin - 1.0)
    deltasmax = np.log10(sigmagmax - 1.0)
    sigmag_arr = np.logspace(deltasmin, deltasmax, Nsigmag) + 1.0
    print("sigmag arr = ", sigmag_arr)
    rg_arr = np.logspace(log_rg_min, log_rg_max, Nrg)  # cm

    miegrid = compute_mie_coeff_lognormal_grid(
        refraction_index,
        refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        N0=N0,
    )
    save_miegrid(filename, miegrid, rg_arr, sigmag_arr)


def evaluate_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr):
    """evaluates the value at rg and sigmag by interpolating miegrid

    Args:
        rg (float): rg parameter in lognormal distribution
        sigmag (float): sigma_g parameter in lognormal distribution
        miegrid (5d array): Mie grid (lognormal)
        sigmag_arr (1d array): sigma_g array
        rg_arr (1d array): rg array

    Note:
        beta derived here is in the unit of 1/Mm (Mega meter) for diameter
        multiply 1.e-8 to convert to 1/cm for radius.


    Returns:
        _type_: evaluated values of miegrid, output of MieQ_lognormal Bext (1/Mm), Bsca, Babs, G, Bpr, Bback, Bratio (wavenumber, number of mieparams)
    """
    # mieparams = interp2d_bilinear(rg, sigmag, rg_arr, sigmag_arr, miegrid)
    mieparams = interp2d_bilinear(
        jnp.log(rg), sigmag, jnp.log(rg_arr), sigmag_arr, miegrid
    )  # interpolation in log space for rg, linear for sigmag
    return mieparams


def compute_mieparams_cgs_from_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr, N0):
    """computes Mie parameters i.e. extinction coeff, sinigle scattering albedo, asymmetric factor using miegrid. This process also convert the unit to cgs

    Args:
        rg_layer (1d array): layer wise rg parameters
        sigmag_layer (1d array): layer wise sigmag parameters
        miegrid (5d array): Mie grid (lognormal)
        rg_arr (1d array): rg array used for computing miegrid
        sigmag_arr (1d array): sigma_g array used for computing miegrid
        N0 (float, optional): the normalization of the lognormal distribution ($N_0$). Defaults to 1.0.

    Note:
        N0 ($N_0$) is defined as the normalization of the cloud distribution,
        $n(r) pr = N_0 p(r) pr
        where $r$ is the particulate radius and
        $p(r)  = \frac{1}{sqrt{2 \pi}} \frac{1}{r \log{\sigma_g}} \exp{-\frac{(\log{r} - \log{r_g})^2}{2 \log^2{\sigma_g}}}$$
        is the lognormal distribution.
        See also https://pymiescatt.readthedocs.io/en/latest/forward.html
        they use the diameter instead (but p(d) )
        $p(d) pd = \frac{N_0}{sqrt{2 \pi}} \frac{1}{d \log{\sigma_g}} \exp{-\frac{(\log{d} - \log{d_g})^2}{2 \log^2{\sigma_g}}}$ pd
        where  and $d = 2 r$ and $d_g = 2 r_g$.
        In ExoJAX Ackerman and Marley cloud model, $N_0$ can be evaluated by `atm.amclouds.normalization_lognormal`.

    Note:
        Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*sigma0_extinction
        The original extinction coefficient (beta) from PyMieScat has the unit of 1/Mm (Mega meter) for diameter.
        Therefore, this method multiplies 1.e-8 to beta for conversion to 1/cm for radius.

    Returns:
        sigma_extinction, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
        sigma_scattering, scattering cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
        g, asymmetric factor (mean g)
    """

    mieparams = evaluate_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr)
    convfactor_to_cgs = 1.0e-8 / N0  # conversion to cgs(1/Mega meter to 1/cm)
    sigma_extinction = convfactor_to_cgs * mieparams[:, 0]  # (layer, wav)
    sigma_scattering = convfactor_to_cgs * mieparams[:, 1]
    g = mieparams[:, 3]

    return sigma_extinction, sigma_scattering, g


def mie_lognormal_pymiescatt(
    m,
    wavelength,
    sigmag,
    rg,
    N0,
    rgrid,
    nMedium=1.0,
):
    """Mie parameters assuming a lognormal distribution

    Args:
        m (_type_): _description_
        wavelength (_type_): _description_
        sigmag (float): sigma_g parameter in lognormal distribution
        rg (float): rg parameter in lognormal distribution in cgs
        N0 (_type_):
        rgrid: grid of the particulate radius
        nMedium (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    from PyMieScatt.Mie import Mie_SD
    from exojax.special.lognormal import pdf

    #  http://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_Lognormal
    nMedium = nMedium.real
    m /= nMedium
    wavelength /= nMedium

    dp = 2.0 * rgrid
    ndp = N0 * pdf(dp, 2.0 * rg, sigmag)

    Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio = Mie_SD(
        m, wavelength, dp, ndp, SMPS=False
    )
    return Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio


def auto_rgrid(rg, sigmag, nrgrid=1500):
    """sets the automatic rgrid (particulate radius grid).

    Note:
        This method optimizes the sampling grid of the particulate radius (rgrid).
        The basic strategy is as follows.
        1. if the cube-weighted lognormal distribution q(x) is enough compact, the sampling points is given by the linear within the range of [mean - m sigma, mean + n sigma]
        where sigma is the STD of q(x).
        2. else, i.e., the distribution starts from close to 0 (large sigmag), the range is [0 + dr, n*mean].
        Currently sigmag = 1.0001 to 4 is within 1 % error for the defalut setting. See tests/unittests/spec/clouds/mie_test.py

    Args:
        rg (float): rg parameter in lognormal distribution in cgs
        sigmag (float): sigma_g parameter in lognormal distribution
        nrgrid (int, optional): the number of rgrid. Defaults to 1500.

    Returns:
        array: rgrid (the particulate radius grid)
    """
    if sigmag <= 1.0:
        raise ValueError("sigamg must be larger than 1.")
    if sigmag < 1.0001 or sigmag > 4.0:
        warnings.warn(
            "May be out of the robust sigmag range (1.0001-4.0). sigmag=" + str(sigmag)
        )

    n = 13
    m = 4
    mu = cubeweighted_mean(rg, sigmag)
    std = cubeweighted_std(rg, sigmag)
    if m * std < mu:
        rgrid_lower = mu - m * std
        rgrid_upper = mu + n * std
        dr = (rgrid_upper - rgrid_lower) / nrgrid
    else:
        dr = n * mu / nrgrid
        rgrid_lower = dr
        rgrid_upper = n * mu

    rgrid = np.linspace(rgrid_lower, rgrid_upper, nrgrid)
    return rgrid


def cubeweighted_integral_checker(rgrid, rg, sigmag, accuracy=1.0e-2):
    """checks if the trapezoid integral of the cube-weighted lognormal distribution agrees with 1 within prec given rgrid, rg, sigmag

    Args:
        rgrid (array): the particulate radius grid
        rg (float): rg parameter in lognormal distribution in cgs
        sigmag (float): sigma_g parameter in lognormal distribution
        accuracy (float, optional): _description_. Defaults to 1.e-2, i.e. 1% accuracy.

    Returns:
        bool: True or False
    """
    val = cubeweighted_pdf(rgrid, rg, sigmag)
    intvalue = trapezoid(val, rgrid)
    error = np.abs(intvalue - 1.0)
    return error < accuracy


if __name__ == "__main__":
    from exojax.spec.pardb import PdbCloud
    import jax.numpy as jnp

    pdb = PdbCloud("NH3")
    filename = ".database/particulates/virga/miegrid_lognorm_" + pdb.condensate + ".mg"
    # make_miegrid_lognormal(
    #    pdb.refraction_index, pdb.refraction_index_wavelength_nm, filename
    # )

    # load
    miegrid, rg_arr, sigmag_arr = read_miegrid(filename + ".npz")
    rg = 3.0e-5
    sigmag = 2.0
    f = evaluate_miegrid(rg, sigmag, miegrid, rg_arr, sigmag_arr)
