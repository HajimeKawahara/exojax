from exojax.utils.constants import hcperk
import jax.numpy as jnp
from jax import vmap
from jax import grad
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pkg_resources


def _beta(t, tref):
    return hcperk * (t - tref)


def weight_point2_dE(t, tref, dE, p=0.5):
    """dE version of the weight at point 2 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        dE (float): envergy interval between points 1 nad 2 (cm-1)
        p (float): between 0 to 1

    Returns:
        weight at point 2
    """

    fac1 = 1.0 - jnp.exp(_beta(t, tref) * p * dE)
    fac2 = jnp.exp(-_beta(t, tref) *
                   (1.0 - p) * dE) - jnp.exp(_beta(t, tref) * p * dE)
    return fac1 / fac2


def weight_point1_dE(t, tref, dE, p=0.5):
    """dE version of the weight at point 1 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        dE (float): envergy interval between points 1 nad 2 (cm-1)
        p (float): between 0 to 1

    Returns:
        weight at point 1
    """
    return 1.0 - weight_point2_dE(t, tref, dE, p)


def single_tilde_line_strength(t, w1, w2, tref, dE, p=0.5):
    """

    Args:
        t (float): inverse temperature
        w1 (_type_): weight at point 1
        w2 (_type_): weight at point 2
        tref (_type_): reference temperature
        dE (_type_): energy interval in cm-1
        p (float, optional): between 0 to 1 Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    v1 = w1 * jnp.exp(_beta(t, tref) * p * dE)
    v2 = w2 * jnp.exp(-_beta(t, tref) * (1.0 - p) * dE)
    return v1 + v2 - 1.0


def single_tilde_line_strength_zeroth(t, twp, tref, dE, p=0.5):
    w1 = weight_point1_dE(twp, tref, dE, p)
    w2 = weight_point2_dE(twp, tref, dE, p)
    return single_tilde_line_strength(t, w1, w2, tref, dE, p)


def single_tilde_line_strength_first(t, twp, tref, dE, p=0.5):
    """Single Line Line strength prediction for Premodit/diffmode=1

    Args:
        t (_type_): inverse temperature
        twp (_type_): inverse weight temperature
        tref (_type_): inverse reference temperature
        dE (_type_): Elower interval
        p (float, optional): fraction of the line point. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    dfw1 = grad(weight_point1_dE, argnums=0)
    dfw2 = grad(weight_point2_dE, argnums=0)
    w1 = weight_point1_dE(twp, tref, dE,
                          p) + dfw1(twp, tref, dE, p) * (t - twp)
    w2 = weight_point2_dE(twp, tref, dE,
                          p) + dfw2(twp, tref, dE, p) * (t - twp)
    return single_tilde_line_strength(t, w1, w2, tref, dE, p)


def single_tilde_line_strength_second(t, twp, tref, dE, p=0.5):
    """Single Line Line strength prediction for Premodit/diffmode=1

    Args:
        t (_type_): inverse temperature
        twp (_type_): inverse weight temperature
        tref (_type_): inverse reference temperature
        dE (_type_): Elower interval
        p (float, optional): fraction of the line point. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    dfw1 = grad(weight_point1_dE, argnums=0)
    dfw2 = grad(weight_point2_dE, argnums=0)
    ddfw1 = grad(dfw1, argnums=0)
    ddfw2 = grad(dfw2, argnums=0)

    w1 = weight_point1_dE(twp, tref, dE, p) + dfw1(twp, tref, dE, p) * (
        t - twp) + ddfw1(twp, tref, dE, p) * (t - twp)**2 / 2.0
    w2 = weight_point2_dE(twp, tref, dE, p) + dfw2(twp, tref, dE, p) * (
        t - twp) + ddfw2(twp, tref, dE, p) * (t - twp)**2 / 2.0
    return single_tilde_line_strength(t, w1, w2, tref, dE, p)


def worst_tilde_line_strength_first(T, Ttyp, Tref, dE):
    """worst deviation of single tilde line search first in terms of p

    Args:
        T (float, ndarray): temperature (array) K
        Twp (float): weight temperature K
        Tref (float): reference tempearture K
        dE (float): Elower interval cm-1

    Return:
        worst value of single_tilde_line_strength_first in terms of p

    """
    def f(p):
        return single_tilde_line_strength_first(1 / T, 1 / Ttyp, 1 / Tref, dE,
                                                p)

    ff = vmap(f)
    parr = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return jnp.max(jnp.abs(ff(parr)), axis=0)


def worst_tilde_line_strength_second(T, Ttyp, Tref, dE):
    """worst deviation of single tilde line search first in terms of p

    Args:
        T (float, ndarray): temperature (array) K
        Twp (float): weight temperature K
        Tref (float): reference tempearture K
        dE (float): Elower interval cm-1

    Return:
        worst value of single_tilde_line_strength_first in terms of p

    """
    def f(p):
        return single_tilde_line_strength_second(1 / T, 1 / Ttyp, 1 / Tref, dE,
                                                 p)

    ff = vmap(f)
    parr = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return jnp.max(jnp.abs(ff(parr)), axis=0)


def evaluate_trange(Tarr, tlide_line_strength, crit, Twt):
    """evaluate robust temperature range

    Args:
        Tarr (ndarray): temperature array, shape = (N,)
        tlide_line_strength (ndarray): line strength error, shape = (N,)
        crit (float): criterion of line strength error (0.01=1%)
        Twt (float): weight temperature
        
    Returns:
        float, float: Tl, Tu. The line strength error is below crit within [Tl, Tu] 
    """

    #exclude all nan case
    validmask = tlide_line_strength == tlide_line_strength
    novalidmask = np.logical_not(validmask)
    if len(tlide_line_strength[validmask]) == 0:
        return Twt, Twt
    elif len(tlide_line_strength[novalidmask]) > 0:
        tlide_line_strength[novalidmask] = np.inf

    abs_tlide_line_strength = np.abs(tlide_line_strength)
    dT = Twt - Tarr
    mask_lower = (abs_tlide_line_strength > crit) * (dT > 0)
    dT_masked_lower = dT[mask_lower]
    if len(dT_masked_lower) > 0:
        Tl = Twt - np.min(dT_masked_lower)
    else:
        Tl = Tarr[0]

    mask_upper = (abs_tlide_line_strength > crit) * (dT < 0)
    dT_masked_upper = dT[mask_upper]
    if len(dT_masked_upper) > 0:
        Tu = Twt + np.min(-dT_masked_upper)
    else:
        Tu = Tarr[-1]
    return Tl, Tu


def default_elower_grid_trange_file():
    """default elower_grid_trange filename
    Returns:
        default_elower_grid_trange filename

    Note:
        This file assumes 1 % precision of line strength within [Tl, Tu]
        default_elower_grid_trange_file generated by 
        examples/gendata/gen_elower_grid_trange.py.
    """
    filename = pkg_resources.resource_filename(
        'exojax', 'data/premodit/elower_grid_trange.npz')
    return filename


def optimal_params(Tl,
                   Tu,
                   diffmode=2,
                   makefig=False,
                   filename=None):
    """derive the optimal parameters for a given Tu and Tl, 
       which satisfies x % (1% if filename=None) precision within [Tl, Tu]

    Args:
        Tl (float): lower temperature
        Tu (float): upper temperature
        diffmode (int, optional): diff mode. Defaults to 2.
        makefig (bool, optional): if you wanna make a fig. Defaults to False.
        filename: grid_trange file,  if None default_elower_grid_trange_file is used.

    
    Returns:
        float: dE, Tref, Twt (optimal ones)
    """

    if filename is None:
        filename = default_elower_grid_trange_file()
    dat = np.load(filename)
    arr = dat["arr_0"]
    #Tarr = dat["arr_1"]
    Twtarr = dat["arr_2"]
    Trefarr = dat["arr_3"]
    dEarr = dat["arr_4"]

    if Tl > Tu:
        raise ValueError("Tl must be smaller than Tu.")
    if diffmode > 2:
        raise ValueError("diffmode is currently to be <= 2.")

    maskl = (arr[0, :, :, :, diffmode] <= Tl)
    masku = (arr[1, :, :, :, diffmode] >= Tu)
    mask = maskl * masku
    for i in range(len(dEarr)):
        k = -i - 1
        j = np.sum(mask[:, :, k])
        Tlarr = arr[0, :, :, k, diffmode]
        Tuarr = arr[1, :, :, k, diffmode]
        if j > 0:
            indices = np.where(mask[:, :, k])
            Twtallow = Twtarr[indices[0]]
            Trefallow = Trefarr[indices[1]]

        if j > 0 and makefig:
            makefig = False
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(131)
            c = _draw_map(Tlarr, ax, Trefarr, Twtarr, Tu * 1.2)
            _pltadd(c, Trefallow, Twtallow)
            ax = fig.add_subplot(132)
            c = _draw_map(Tuarr, ax, Trefarr, Twtarr, Tu * 1.2)
            _pltadd(c, Trefallow, Twtallow)
            ax = fig.add_subplot(133)
            c = _draw_map(Tuarr - Tlarr, ax, Trefarr, Twtarr, Tu * 1.2)
            _pltadd(c, Trefallow, Twtallow)
            plt.show()

        if j == 1:
            return dEarr[k]*(diffmode+1), Trefallow[0], Twtallow[0]
        elif j > 1:
            #choose the largest interval.        
            Tlx = arr[0, indices[0], indices[1], k, diffmode]
            Tux = arr[1, indices[0], indices[1], k, diffmode]
            dT = (Tux-Tlx)
            ind = np.argsort(dT)[::-1]
            print("Robust range:",Tlx[ind[0]],"-",Tux[ind[0]],"K")
            return dEarr[k]*(diffmode+1), Trefallow[ind[0]], Twtallow[ind[0]]

        if i == len(dEarr) - 1:
            warnings.warn("Couldn't find the params.")
            return None, None, None


def _draw_map(value, ax, Trefarr, Twtarr, Tmax_view):
    c = ax.pcolor(Trefarr,
                  Twtarr,
                  value,
                  cmap="rainbow",
                  vmin=0.0,
                  vmax=Tmax_view)
    ax.set_aspect("equal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return c


def _pltadd(c, Trefallow, Twtallow):
    plt.plot((Trefallow), (Twtallow), "+", color="white")
    plt.colorbar(c, shrink=0.7)
    plt.ylabel("Twt (K)")
    plt.xlabel("Tref (K)")
