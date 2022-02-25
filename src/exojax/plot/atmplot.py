"""plotting tool for atmospheric structure."""
import numpy as np
import matplotlib.pyplot as plt


def plottau(nus, dtauM, Tarr=None, Parr=None, unit=None, mode=None, vmin=-3, vmax=3):
    """Plot optical depth (tau). This function gives the color map of log10(tau) (or log10(dtau)), optionally w/ a T-P profile.

    Args:
       nus: wavenumber
       dtauM: dtau matrix
       Tarr: temperature profile
       Parr: perssure profile
       unit: x-axis unit=um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom),
       mode: mode=None (lotting tau), mode=dtau (plotting delta tau for each layer)
       vmin: color value min (default=-3)
       vmax: color value max (default=3)
    """
    if mode == 'dtau':
        ltau = np.log10(dtauM)
    else:
        ltau = np.log10(np.cumsum(dtauM, axis=0))

    plt.figure(figsize=(20, 3))
    ax = plt.subplot2grid((1, 20), (0, 3), colspan=18)
    if unit == 'um':
        c = ax.imshow(ltau[:, ::-1], vmin=vmin, vmax=vmax, cmap='RdYlBu_r', alpha=0.9,
                      extent=[1.e4/nus[-1], 1.e4/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
        plt.xlabel('wavelength ($\mu \mathrm{m}$)')
    elif unit == 'nm':
        c = ax.imshow(ltau[:, ::-1], vmin=vmin, vmax=vmax, cmap='RdYlBu_r', alpha=0.9,
                      extent=[1.e7/nus[-1], 1.e7/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
        plt.xlabel('wavelength (nm)')
    elif unit == 'AA':
        c = ax.imshow(ltau[:, ::-1], vmin=vmin, vmax=vmax, cmap='RdYlBu_r', alpha=0.9,
                      extent=[1.e8/nus[-1], 1.e8/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
        plt.xlabel('wavelength ($\AA$)')
    else:
        c = ax.imshow(ltau, vmin=vmin, vmax=vmax, cmap='RdYlBu_r', alpha=0.9, extent=[
                      nus[0], nus[-1], np.log10(Parr[-1]), np.log10(Parr[0])])
        plt.xlabel('wavenumber ($\mathrm{cm}^{-1}$)')

    plt.colorbar(c, shrink=0.8)
    plt.ylabel('log10 (P (bar))')
    ax.set_aspect(0.2/ax.get_data_ratio())
    if Tarr is not None and Parr is not None:
        ax = plt.subplot2grid((1, 20), (0, 0), colspan=2)
        plt.plot(Tarr, np.log10(Parr), color='gray')
        plt.xlabel('temperature (K)')
        plt.ylabel('log10 (P (bar))')
        plt.gca().invert_yaxis()
        plt.ylim(np.log10(Parr[-1]), np.log10(Parr[0]))
        ax.set_aspect(1.45/ax.get_data_ratio())


def plotcf(nus, dtauM, Tarr, Parr, dParr, unit=None, mode=None, log=False, normalize=True, cmap='bone_r'):
    """plot the contribution function. This function gives a plot of contribution function, optionally w/ a T-P profile.

    Args:
       nus: wavenumber
       dtauM: dtau matrix
       Tarr: temperature profile
       Parr: perssure profile
       dParr: perssure difference profile
       unit: x-axis unit=um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom),
       mode: None=contour, "cmap"=color map
       log: True=use log10(cf)
       normalize: normalize cf for each wavenumber?
       cmap: colormap

    Returns:
       contribution function
    """
    from exojax.spec.planck import piBarr
    hcperk = 1.4387773538277202
    tau = np.cumsum(dtauM, axis=0)

    cf = np.exp(-tau)*dtauM\
        * (Parr[:, None]/dParr[:, None])\
        * nus**3/(np.exp(hcperk*nus/Tarr[:, None])-1.0)

    if normalize == True:
        cf = (cf/np.sum(cf, axis=0))
    if log == True:
        cf = np.log10(cf)

    plt.figure(figsize=(20, 3))
    ax = plt.subplot2grid((1, 20), (0, 3), colspan=18)
    if mode == 'cmap':
        if unit == 'um':
            c = ax.imshow(cf[:, ::-1], cmap=cmap, alpha=0.9, extent=[1.e4 /
                          nus[-1], 1.e4/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
            plt.xlabel('wavelength ($\mu \mathrm{m}$)')
        elif unit == 'nm':
            c = ax.imshow(cf[:, ::-1], cmap=cmap, alpha=0.9, extent=[1.e7 /
                          nus[-1], 1.e7/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
            plt.xlabel('wavelength (nm)')
        elif unit == 'AA':
            c = ax.imshow(cf[:, ::-1], cmap=cmap, alpha=0.9, extent=[1.e8 /
                          nus[-1], 1.e8/nus[0], np.log10(Parr[-1]), np.log10(Parr[0])])
            plt.xlabel('wavelength ($\AA$)')
        else:
            c = ax.imshow(cf, cmap=cmap, alpha=0.9, extent=[
                          nus[0], nus[-1], np.log10(Parr[-1]), np.log10(Parr[0])])
            plt.xlabel('wavenumber ($\mathrm{cm}^{-1}$)')
    else:
        if unit == 'um':
            X, Y = np.meshgrid(1.e4/nus, np.log10(Parr))
            plt.xlabel('wavelength ($\mu \mathrm{m}$)')
        elif unit == 'nm':
            X, Y = np.meshgrid(1.e7/nus, np.log10(Parr))
            plt.xlabel('wavelength (nm)')
        elif unit == 'AA':
            X, Y = np.meshgrid(1.e8/nus, np.log10(Parr))
            plt.xlabel('wavelength ($\AA$)')
        else:
            X, Y = np.meshgrid(nus, np.log10(Parr))
            plt.xlabel('wavenumber ($\mathrm{cm}^{-1}$)')

        c = ax.contourf(X, Y, cf, 30, cmap=cmap)
        plt.gca().invert_yaxis()

    plt.ylabel('log10 (P (bar))')
    plt.colorbar(c, shrink=0.8)
    ax.set_aspect(0.2/ax.get_data_ratio())

    if Tarr is not None and Parr is not None:
        ax = plt.subplot2grid((1, 20), (0, 0), colspan=2)
        plt.plot(Tarr, np.log10(Parr), color='gray')
        plt.xlabel('temperature (K)')
        plt.ylabel('log10 (P (bar))')
        plt.gca().invert_yaxis()
        plt.ylim(np.log10(Parr[-1]), np.log10(Parr[0]))
        ax.set_aspect(1.45/ax.get_data_ratio())

    return cf


def plot_maxpoint(mask, Parr, maxcf, maxcia, mol='CO'):
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
    masknon0 = (maxcf > 0)
    plt.plot(xarr[masknon0], Parr[maxcf[masknon0]], '.',
             label=mol, alpha=1.0, color='gray', rasterized=True)
    plt.plot(xarr[mask], Parr[maxcf[mask]], '.', label=mol +
             ' selected', alpha=1.0, color='C3', rasterized=True)
    plt.plot(xarr, Parr[maxcia], '-', label='CIA (H2-H2)',
             alpha=1.0, color='C2', rasterized=True)

    plt.yscale('log')
    plt.ylim(Parr[0], Parr[-1])
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=20)
    plt.xlabel('#line', fontsize=20)
    plt.ylabel('Pressure (bar)', fontsize=20)
    plt.legend(fontsize=20)
