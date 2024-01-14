"""elower grid trange file
"""

import numpy as np
import tqdm
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import worst_tilde_line_strength_first
from exojax.spec.lbderror import worst_tilde_line_strength_second
from exojax.spec.lbderror import evaluate_trange


def generate_elower_grid_trange(
    N_Twt,
    N_Tref,
    N_Trange,
    N_dE,
    Treflow,
    Trefhigh,
    Twtlow,
    Twthigh,
    Tsearchl,
    Tsearchh,
    dErangel,
    dErangeh,
    filename,
    precision_criterion=0.01,
):
    """
    generates "elower_grid_trange" file, i.e. robust temperature range (Tlow, Thigh) as a function of dE, Tre, Twt, (and diffmode)

    Args:
        N_Twt (int): the number of Twt grid
        N_Tref (int): the number of Tref grid
        N_Trange (int): the number of Trange to serach for Tlow and Thigh
        N_dE (int): the number of dE grid
        Treflow (float): lower limit of Tref grid
        Trefhigh (float): upper limit of Tref grid
        Twtlow (float): lower limit of Twt grid
        Twthigh (float): upper limit of Twt grid
        Tsearchl (float): lower limit of Trange grid
        Tsearchh (float): upper limit of Trange grid
        dErangel (float): lower limit of dE grid
        dErangeh (float): upper limit of dE grid
        filename (str): output filename
        precision_criterion (float, optional): _description_. Defaults to 0.01.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    Ndiffmode = 3  # diffmode=0,1,2

    # Avoiding Twt = Tref
    if Treflow == Twtlow:
        raise ValueError("Treflow should be slightly diffrent from Twtlow.")
    if Trefhigh == Twthigh:
        raise ValueError("Trefhigh should be slightly diffrent from Twthigh.")

    Twtarr = np.logspace(np.log10(Twtlow), np.log10(Twthigh), N_Twt)
    Trefarr = np.logspace(np.log10(Treflow), np.log10(Trefhigh), N_Tref)
    Tarr = np.logspace(np.log10(Tsearchl), np.log10(Tsearchh), N_Trange)
    dEarr = np.linspace(dErangel, dErangeh, N_dE)
    arr = np.zeros((2, N_Twt, N_Tref, N_dE, Ndiffmode))
    for idE, dE in enumerate(tqdm.tqdm(dEarr)):
        for iTref, Tref in enumerate(Trefarr):
            for iTwt, Twt in enumerate(Twtarr):
                x = single_tilde_line_strength_zeroth(
                    1.0 / Tarr, 1.0 / Twt, 1.0 / Tref, dE
                )
                Tl, Tu = evaluate_trange(Tarr, x, precision_criterion, Twt)
                arr[0, iTwt, iTref, idE, 0] = Tl
                arr[1, iTwt, iTref, idE, 0] = Tu

                x = worst_tilde_line_strength_first(Tarr, Twt, Tref, dE * 2)
                Tl, Tu = evaluate_trange(Tarr, x, precision_criterion, Twt)
                arr[0, iTwt, iTref, idE, 1] = Tl
                arr[1, iTwt, iTref, idE, 1] = Tu

                x = worst_tilde_line_strength_second(Tarr, Twt, Tref, dE * 3)
                Tl, Tu = evaluate_trange(Tarr, x, precision_criterion, Twt)
                arr[0, iTwt, iTref, idE, 2] = Tl
                arr[1, iTwt, iTref, idE, 2] = Tu

    np.savez(filename, arr, Tarr, Twtarr, Trefarr, dEarr)


def params_version1():
    """
    parameters for the defaut elower grid trange version 1

    Returns:
        params for generate_elower_grid_trange
    """
    N_Twt = 50
    N_Tref = N_Twt
    N_Trange = 120
    N_dE = 29
    Trefl = 100.0
    Trangelwt = Trefl + 0.1
    Trefh = 2000.0
    Trangehwt = Trefh + 0.1
    Tsearchl = 100.0
    Tsearchh = 5000.0
    dErangel = 100
    dErangeh = 1500
    filename = "elower_grid_trange.npz"
    return (
        N_Twt,
        N_Tref,
        N_Trange,
        N_dE,
        Trefl,
        Trefh,
        Trangelwt,
        Trangehwt,
        Tsearchl,
        Tsearchh,
        dErangel,
        dErangeh,
        filename,
    )


def params_version2():
    """
    parameters for the defaut elower grid trange version 2

    Returns:
        params for generate_elower_grid_trange
    """
    N_Twt = 75
    N_Tref = N_Twt
    N_Trange = 120
    N_dE = 39
    Trefl = 50.0
    Trangelwt = Trefl + 0.1
    Trefh = 3000.0
    Trangehwt = Trefh + 0.1
    Tsearchl = 50.0
    Tsearchh = 7500.0
    dErangel = 50
    dErangeh = 1000
    filename = "elower_grid_trange_v2.npz"
    return (
        N_Twt,
        N_Tref,
        N_Trange,
        N_dE,
        Trefl,
        Trefh,
        Trangelwt,
        Trangehwt,
        Tsearchl,
        Tsearchh,
        dErangel,
        dErangeh,
        filename,
    )


if __name__ == "__main__":
    (
        N_Twt,
        N_Tref,
        N_Trange,
        N_dE,
        Trefl,
        Trefh,
        Trangelwt,
        Trangehwt,
        Tsearchl,
        Tsearchh,
        dErangel,
        dErangeh,
        filename,
    ) = params_version2()

    generate_elower_grid_trange(
        N_Twt,
        N_Tref,
        N_Trange,
        N_dE,
        Trefl,
        Trefh,
        Trangelwt,
        Trangehwt,
        Tsearchl,
        Tsearchh,
        dErangel,
        dErangeh,
        filename,
    )
