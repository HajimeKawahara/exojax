import numpy as np
import matplotlib.pyplot as plt


def plot_lbd(
    lbd_coeff,
    elower_grid,
    ngamma_ref_grid,
    n_Texp_grid,
    multi_index_uniqgrid,
    nu_grid,
    vmin=-70,
    vmax=-20,
    order=0,
    number_of_ticks = 10
):
    """Plots the line basis density

    Note:
        See #548 for more details.

    Examples:
        nu_grid, wav, resolution = wavenumber_grid(
            22900.0, 27000.0, 200000, unit="AA", xsmode="premodit"
        )
        mdb = MdbExomol(".database/CO/12C-16O/Li2015/", nurange=nu_grid)
        opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1000.0], dit_grid_resolution=0.2)
        lbd, midx, gegrid, gngamma, gn, R, pm = opa.opainfo
        plot_lbd(lbd, gegrid, gngamma, gn, midx, nu_grid)


    Args:
        lbd_coeff (array): line basis density coefficients
        elower_grid (array): elower grid
        ngamma_ref_grid (array): n_gamma_ref grid
        n_Texp_grid (array): n_Texp grid
        multi_index_uniqgrid (array): multi index grid
        nu_grid (array): wavenumber grid
        vmin (int, optional): min value of color. Defaults to -70.
        vmax (int, optional): max value of color. Defaults to -20.
        order (int, optional): order of the LBD coefficient. Defaults to 0.
        number_of_ticks (int, optional): number of ticks. Defaults to 10.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter

    lbd = np.exp(lbd_coeff[order, :, :, :])
    arr = np.log10(np.nansum(lbd, axis=1))  # integrates over broadening parameters
    arrx = np.log10(np.nansum(lbd, axis=0))  # integrate over Elower
    n = int(len(nu_grid) / number_of_ticks)
    log_ticks = np.log10(nu_grid[::n])

    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    ax = fig.add_subplot(gs[0, :4])
    ax.set_xticks(log_ticks)

    # Warning: interpolation = "none" in imshow is very important, otherwise the fine structure is washed out.
    extent = [np.log10(nu_grid[0]),np.log10(nu_grid[-1]),elower_grid[-1],elower_grid[0]]
    c = _lbd_imshow(extent, vmin, vmax, arr, ax)
    cbar = plt.colorbar(c)
    cbar.set_label("log10(LBD (cm/bin))")
    ax.xaxis.set_major_formatter(FuncFormatter(_log_formatter))
    ax.set_xlabel("wavenumber $\,(\mathrm{cm}^{-1})$")
    ax.set_ylabel("$E \, (\mathrm{cm}^{-1})$")
    plt.gca().invert_yaxis()

    ax = fig.add_subplot(gs[0, 4])
    extent=[0, len(multi_index_uniqgrid) - 1, elower_grid[-1], elower_grid[0]]
    c = _lbd_imshow(extent, vmin, vmax, arrx, ax)
    ax.xaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    _set_xlabel_with_range(ngamma_ref_grid, ax, "width", 1)
    ax2 = ax.twiny()
    ax2.axes.get_xaxis().set_ticks([])
    _set_xlabel_with_range(n_Texp_grid, ax2, "power", 3)

    cbar = plt.colorbar(c)
    cbar.set_label("log10(LBD (cm/bin))")
    _put_braodening_indices(elower_grid, multi_index_uniqgrid, ax)
    ax.set_ylabel("$E \, (\mathrm{cm}^{-1})$")
    plt.gca().invert_yaxis()

def _lbd_imshow(extent, vmin, vmax, arr, ax):
    c = ax.imshow(
        arr.T,
        aspect="auto",
        cmap="inferno",
        interpolation="none",
        extent = extent,
        vmin=vmin,
        vmax=vmax,
    )
    
    return c

def _put_braodening_indices(elower_grid, multi_index_uniqgrid, ax):
    for i, miu in enumerate(multi_index_uniqgrid):
        iwidth = miu[0]
        ipower = miu[1]
        ax.text(i, elower_grid[0], str(iwidth), ha="center", va="top")
        ax.text(i, elower_grid[-1], str(ipower), ha="center", va="bottom")


def _log_formatter(value, tick_number):
    return f"{10**value:.1f}"


def _set_xlabel_with_range(grid_for_label, ax, lab, decimals):
    ax.set_xlabel(
        "# for "
        + lab
        + ": "
        + str(np.round(grid_for_label[0], decimals))
        + " - "
        + str(np.round(grid_for_label[-1], decimals))
        + " cm-1",
        labelpad=12,
    )


def plot_broadening_parameters_grids(
    ngamma_ref_grid,
    n_Texp_grid,
    nu_grid,
    resolution,
    gamma_ref_in,
    n_Texp_in,
    crit,
    figname,
):
    if 2 * len(gamma_ref_in) > crit:
        n = int(len(gamma_ref_in) / crit)
        gamma_ref = gamma_ref_in[::n]
        n_Texp = n_Texp_in[::n]
        print("Show broadening parameter sampling points, n_sample=", str(n))
        print(len(gamma_ref_in), "->", len(gamma_ref), "pts")
    else:
        gamma_ref = gamma_ref_in
        n_Texp = n_Texp_in

    maxgamma = ngamma_ref_grid / resolution * np.max(nu_grid)
    gammag_max, n_Texpg_max = _mesh_grid(n_Texp_grid, maxgamma)

    mingamma = ngamma_ref_grid / resolution * np.min(nu_grid)
    gammag_min, n_Texpg_min = _mesh_grid(n_Texp_grid, mingamma)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(gamma_ref, n_Texp, ".", alpha=0.3, label="data")
    plt.plot(gammag_min, n_Texpg_min, "x", label="grid (min)", color="gray")
    plt.plot(gammag_max, n_Texpg_max, "+", label="grid (max)", color="black")

    plt.xscale("log")
    plt.xlabel("$\gamma_\mathrm{ref}$  $(\mathrm{cm}^{-1})$")
    plt.ylabel("temperature exponent")
    plt.legend()
    plt.savefig(figname)
    plt.show()


def _mesh_grid(n_Texp_grid, gamma):
    X, Y = np.meshgrid(gamma, n_Texp_grid)
    grid = np.column_stack([X.flatten(), Y.flatten()])
    gammag = grid[:, 0]
    n_Texpg = grid[:, 1]
    return gammag, n_Texpg
