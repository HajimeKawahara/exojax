import numpy as np
from exojax.spec.premodit import xsvector_zeroth
from exojax.spec.opacalc import OpaPremodit
from exojax.utils.constants import Tref_original
from exojax.spec import normalized_doppler_sigma
from exojax.utils.constants import Tref_original
from tqdm import tqdm


def optelower(mdb,
              nu_grid,
              Tmax,
              Pmin,
              accuracy=0.01,
              dE=100.0,
              display=False):
    """look for the value of the optimal maximum Elower 

    Note:
        The memory use of PreModit depends on the maximum Elower of mdb. 
        This function determine the optimal maximum Elower that does not change 
        the cross section within the accuracy. 

    Args:
        mdb (mdb): molecular db
        nu_grid (array): wavenumber array cm-1
        Tmax (float): maximum temperature in your use (K)
        Pmin (float): minimum temperature in your use (bar)
        accuracy (float, optional): accuracy allowed. Defaults to 0.01.
        dE (float, optional): E grid to search for the optimal Elower. Defaults to 100.0.
        display (bool, optional): if you want to compare the cross section using Eopt w/ ground truth, set True. Defaults to False.

    Returns:
        float: optimal maximum Elower (Eopt) in cm-1 
    """
    from jax.config import config
    config.update("jax_enable_x64", True)
    print("Maximum Elower = ",np.max(mdb.elower))

    #for single temperature, 0-th order is sufficient
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, diffmode=0)
    opa.manual_setting(dE, Tref_original, Tmax)
    lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    nsigmaD = normalized_doppler_sigma(Tmax, mdb.molmass, R)
    qt = mdb.qr_interp(Tmax)

    Tref_broadening = Tref_original
    xsv_master = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff, Tref_original,
                                 R, pmarray, opa.nu_grid, elower_grid,
                                 multi_index_uniqgrid, ngamma_ref_grid,
                                 n_Texp_grid, qt, Tref_broadening)
    allow = True
    q = -1
    pbar = tqdm(total=len(elower_grid), desc="opt Emax")
    while allow:
        xsv = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff[:, :, :, :q],
                              Tref_original, R, pmarray, opa.nu_grid,
                              elower_grid[:q], multi_index_uniqgrid,
                              ngamma_ref_grid, n_Texp_grid, qt, Tref_broadening)
        maxdelta = np.max(np.abs(xsv / xsv_master - 1.0))
        if maxdelta > accuracy:
            allow = False
            q = q + 1
        else:
            q = q - 1
        pbar.update(1)
    pbar.close()
    if q == 0:
        return np.max(mdb.elower)
    if display:
        xsv = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff[:, :, :, :q],
                              Tref_original, R, pmarray, opa.nu_grid,
                              elower_grid[:q], multi_index_uniqgrid,
                              ngamma_ref_grid, n_Texp_grid, qt, Tref_broadening)
        _plot_comparison(nu_grid, xsv_master, xsv)
    Emax = elower_grid[:q][-1]
    return Emax


def _plot_comparison(nu_grid, xsv_master, xsv):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(nu_grid, xsv_master)
    plt.plot(nu_grid, xsv)
    plt.yscale("log")
    ax = fig.add_subplot(212)
    plt.plot(nu_grid, xsv / xsv_master - 1.0)
    plt.ylim(-0.03, 0.03)
    plt.show()
