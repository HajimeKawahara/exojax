"""Check tool of the wavenumber grid."""

import numpy as np
from exojax.utils.constants import c


def warn_resolution(resolution, crit=700000.0):
    """warning poor resolution.

    Args:
        resolution: spectral resolution
        crit: critical resolution
    """
    if resolution < crit:
        print('WARNING: resolution may be too small. R=', resolution)


def check_scale_xsmode(xsmode):
    """checking if the scale of xsmode assumes ESLOG(log) or ESLIN(linear)

    Args:
       xsmode: xsmode

    Return:
       ESLOG/ESLIN/UNKNOWN
    """

    if xsmode == 'lpf' or xsmode == 'LPF' or xsmode == 'modit' or xsmode == 'MODIT' or xsmode == 'redit' or xsmode == 'REDIT':
        print('xsmode assumes ESLOG: mode=', xsmode)
        return 'ESLOG'
    elif xsmode == 'dit' or xsmode == 'DIT':
        print('xsmode assumes ESLIN: mode=', xsmode)
        return 'ESLIN'
    else:
        return 'UNKNOWN'


def check_scale_nugrid(nus, crit1=1.e-5, crit2=1.e-14, gridmode='ESLOG'):
    """checking if nugrid is evenly spaced in a logarithm scale (ESLOG) or a
    liner scale (ESLIN)

    Args:
       nus: wavenumber grid
       crit1: criterion for the maximum deviation of log10(nu)/median(log10(nu)) from ESLOG
       crit2: criterion for the maximum deviation of log10(nu) from ESLOG
       gridmode: ESLOG or ESLIN

    Returns:
       True (nugrid is ESLOG) or False (not)
    """
    if gridmode == 'ESLOG':
        q = np.log10(nus)
    elif gridmode == 'ESLIN':
        q = nus
    p = q[1:]-q[:-1]
    w = (p-np.mean(p))
    if gridmode == 'ESLOG':
        val1 = np.max(np.abs(w))/np.median(p)
        val2 = np.max(np.abs(w))
    elif gridmode == 'ESLIN':
        val1 = np.abs(np.max(np.abs(w))/np.median(p))
        val2 = -np.inf  # do not check
    if val1 < crit1 and val2 < crit2:
        return True
    else:
        return False


if __name__ == '__main__':
    from exojax.spec.unitconvert import wav2nu
    from exojax.spec.rtransfer import nugrid

    wav = np.linspace(22920.23000, 1000)
    nus = wav2nu(wav, inputunit='AA')
    print(check_scale_nugrid(nus, gridmode='ESLIN'))
    print('----------------------')

    nus, wav, res = nugrid(22999, 23000, 1000, 'AA')
    print(check_scale_nugrid(nus))
    nus, wav, res = nugrid(22999, 23000, 10000, 'AA')
    print(check_scale_nugrid(nus))
    nus, wav, res = nugrid(22999, 23000, 100000, 'AA')
    print(check_scale_nugrid(nus))

    nus = np.linspace(1.e8/23000., 1.e8/22999., 1000)
    print(check_scale_nugrid(nus))
    nus = np.linspace(1.e8/23000., 1.e8/22999., 10000)
    print(check_scale_nugrid(nus))
    nus = np.linspace(1.e8/23000., 1.e8/22999., 100000)
    print(check_scale_nugrid(nus))
