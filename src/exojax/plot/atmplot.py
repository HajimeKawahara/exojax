"""plotting tool for atmospheric structure."""

import numpy as np
import matplotlib.pyplot as plt
from exojax.utils.constants import hcperk
from exojax.utils.grids import check_eslog_wavenumber_grid
import warnings


def plottau(nugrid, dtauM, Tarr=None, Parr=None, unit=None, mode=None, vmin=-3, vmax=3):
    """Plot optical depth (tau). This function gives the color map of log10(tau) (or log10(dtau)), optionally w/ a T-P profile.

    Args:
        nus: wavenumber in linear scale
        dtauM: dtau matrix
        Tarr: temperature profile
        Parr: perssure profile
        unit: x-axis unit=um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom),
        mode: mode=None (lotting tau), mode=dtau (plotting delta tau for each layer)
        vmin: color value min (default=-3)
        vmax: color value max (default=3)
    """
    if check_eslog_wavenumber_grid(nugrid):
        warnings.warn(
            "nugrid looks in log scale, results in a wrong X-axis value. Use log10(nugrid) instead."
        )

    if mode == "dtau":
        ltau = np.log10(dtauM)
    else:
        ltau = np.log10(np.cumsum(dtauM, axis=0))

    plt.figure(figsize=(20, 3))
    ax = plt.subplot2grid((1, 20), (0, 3), colspan=18)
    c = None
    if unit == "um" or unit == "nm" or unit == "AA":
        factor, labelx = factor_labelx_for_unit()
        if Parr is not None:
            extent = [
                factor[unit] / nugrid[-1],
                factor[unit] / nugrid[0],
                np.log10(Parr[-1]),
                np.log10(Parr[0]),
            ]
            c = imshow_custom(vmin, vmax, ltau[:, ::-1], extent, ax)
        plt.xlabel(labelx[unit])
    else:
        if Parr is not None:
            extent = [nugrid[0], nugrid[-1], np.log10(Parr[-1]), np.log10(Parr[0])]
            c = imshow_custom(vmin, vmax, ltau, extent, ax)
        plt.xlabel("wavenumber ($\mathrm{cm}^{-1}$)")

    if c is not None:
        plt.colorbar(c, shrink=0.8)
    plt.ylabel("log10 (P (bar))")
    ax.set_aspect(0.2 / ax.get_data_ratio())

    if Tarr is not None and Parr is not None:
        _plot_profile_left_panel(Tarr, Parr, xlabel="temperature (K)")


def factor_labelx_for_unit():
    factor = {}
    factor["um"] = 1.0e4
    factor["nm"] = 1.0e7
    factor["AA"] = 1.0e8
    labelx = {}
    labelx["um"] = "wavelength ($\mu \mathrm{m}$)"
    labelx["nm"] = "wavelength (nm)"
    labelx["AA"] = "wavelength ($\AA$)"
    labelx["cm-1"] = "wavenumber ($\mathrm{cm}^{-1}$)"
    return factor, labelx


def plotcf(
    nus,
    dtauM,
    Tarr,
    Parr,
    dParr,
    unit="cm-1",
    log=False,
    normalize=True,
    cmap="bone_r",
    optarr=None,
    leftxlabel="Warning: specify leftxpanel",
    leftxlog=False,
):
    """plot the contribution function. This function gives a plot of contribution function, optionally w/ a T-P (or any) profile.

    Args:
        nus: wavenumber
        dtauM: dtau matrix
        Tarr: temperature profile
        Parr: perssure profile
        dParr: perssure difference profile
        unit: x-axis unit=cm-1, um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom),
        log: True=use log10(cf)
        normalize: normalize cf for each wavenumber?
        cmap: colormap
        optarr: the optional profile X, i.e. a X-P profile of the left panel, instead of a T-P profile
        leftxlabel: xlable of the left panel
        leftxlog: if True, log for the x grid of the left panel

    Returns:
        contribution function
    """
    tau = np.cumsum(dtauM, axis=0)
    cf = (
        np.exp(-tau)
        * dtauM
        * (Parr[:, None] / dParr[:, None])
        * nus**3
        / np.expm1(hcperk * nus / Tarr[:, None])
        # / (np.exp(hcperk * nus / Tarr[:, None]) - 1.0)
    )

    if normalize:
        cf = cf / np.sum(cf, axis=0)
    if log:
        cf = np.log10(cf)

    plt.figure(figsize=(20, 3))
    ax = plt.subplot2grid((1, 20), (0, 3), colspan=18)
    factor, labelx = factor_labelx_for_unit()

    if unit == "um" or unit == "nm" or unit == "AA":
        xnus = factor[unit] / nus
    else:
        xnus = nus

    X, Y = np.meshgrid(xnus, np.log10(Parr))
    c = ax.contourf(X, Y, cf, 30, cmap=cmap)
    plt.gca().invert_yaxis()

    plt.xlabel(labelx[unit])
    plt.ylabel("log10 (P (bar))")
    plt.colorbar(c, shrink=0.8)
    ax.set_aspect(0.2 / ax.get_data_ratio())

    if optarr is not None and Parr is not None:
        xlabel = leftxlabel
        xlog = leftxlog
    elif Tarr is not None and Parr is not None:
        xlabel = "temperature (K)"
        xlog = False

    _plot_profile_left_panel(Tarr, Parr, xlabel=xlabel, xlog=xlog)

    return cf


def imshow_custom(vmin, vmax, ltau, extent, ax):
    c = ax.imshow(ltau, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", alpha=0.9, extent=extent)
    return c


def _plot_profile_left_panel(Tarr, Parr, xlabel, xlog=False):
    ax = plt.subplot2grid((1, 20), (0, 0), colspan=2)
    plt.plot(Tarr, np.log10(Parr), color="gray")
    plt.xlabel(xlabel)
    plt.ylabel("log10 (P (bar))")
    plt.gca().invert_yaxis()
    plt.ylim(np.log10(Parr[-1]), np.log10(Parr[0]))
    if xlog:
        plt.xscale("log")

    ax.set_aspect(1.45 / ax.get_data_ratio())


def plot_maxpoint(mask, Parr, maxcf, maxcia, mol="CO"):
    """Plotting max contribution function  points.

    Args:
        mask: weak line mask
        Parr: Paressure array
        maxcf: max contribution function of the molecules
        maxcia: max contribution function of CIA
        mol: molecular name
    """
    plt.figure(figsize=(14, 6))
    xarr = np.array(range(0, len(mask)))
    masknon0 = maxcf > 0
    plt.plot(
        xarr[masknon0],
        Parr[maxcf[masknon0]],
        ".",
        label=mol,
        alpha=1.0,
        color="gray",
        rasterized=True,
    )
    plt.plot(
        xarr[mask],
        Parr[maxcf[mask]],
        ".",
        label=mol + " selected",
        alpha=1.0,
        color="C3",
        rasterized=True,
    )
    plt.plot(
        xarr,
        Parr[maxcia],
        "-",
        label="CIA (H2-H2)",
        alpha=1.0,
        color="C2",
        rasterized=True,
    )

    plt.yscale("log")
    plt.ylim(Parr[0], Parr[-1])
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=20)
    plt.xlabel("#line", fontsize=20)
    plt.ylabel("Pressure (bar)", fontsize=20)
    plt.legend(fontsize=20)
