import numpy as np
from exojax.spec.unitconvert import nu2wav, wav2nu
from exojax.spec.check_nugrid import check_scale_xsmode, warn_resolution
from exojax.utils.instfunc import resolution_eslog, resolution_eslin


def gen_wavenumber_grid(x0, x1, N, unit='cm-1', xsmode='lpf'):
    """generating the recommended wavenumber grid based on the cross section
    computation mode.

    Args:
        x0: start wavenumber (cm-1) or wavelength (nm) or (AA)
        x1: end wavenumber (cm-1) or wavelength (nm) or (AA)
        N: the number of the wavenumber grid
        unit: unit of the input grid
        xsmode: cross section computation mode (lpf, dit, modit, premodit)

    Returns:
        wavenumber grid evenly spaced in log space
        corresponding wavelength grid
        resolution
    """
    if check_scale_xsmode(xsmode) == 'ESLOG':
        if unit == 'cm-1':
            nus = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
            wav = nu2wav(nus)
        elif unit == 'nm' or unit == 'AA':
            wav = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
            nus = wav2nu(wav, unit)

        resolution = resolution_eslog(nus)
        warn_resolution(resolution)

    elif check_scale_xsmode(xsmode) == 'ESLIN':
        if unit == 'cm-1':
            nus = np.linspace((x0), (x1), N, dtype=np.float64)
            wav = nu2wav(nus)
        elif unit == 'nm' or unit == 'AA':
            cx1, cx0 = wav2nu(np.array([x0, x1]), unit)
            nus = np.linspace((cx0), (cx1), N, dtype=np.float64)
            wav = nu2wav(nus, unit)

        minr, resolution, maxr = resolution_eslin(nus)
        warn_resolution(minr)

    return nus, wav, resolution
