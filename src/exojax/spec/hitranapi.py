"""API for HITRAN and HITEMP outside HAPI."""
import numpy as np
from radis.db.classes import get_molecule
from radis.db.classes import get_molecule_identifier
import jax.numpy as jnp
from contextlib import redirect_stdout
import os
import warnings
with redirect_stdout(open(os.devnull, 'w')):
    import hapi


def search_molecid(molec):
    """molec id from Hitran/Hitemp filename or molecule name or moleid itself.

    Args:
       molec: Hitran/Hitemp filename or molecule name or molec id itself.

    Return:
       int: molecid (HITRAN molecular id)
    """
    try:
        hitf = molec.split('_')
        molecid = int(hitf[0])
        return molecid
    except:
        try:
            return get_molecule_identifier(molec)
        except:
            try:
                return get_molecule_identifier(get_molecule(molec))
            except:
                raise ValueError(
                    'Not valid HITRAN/Hitemp file or molecular id or molecular name.'
                )


def get_pf(M, I_list):
    """HITRAN/HITEMP IO for partition function

    Args:
        M: HITRAN molecule number
        I_list: HITRAN isotopologue number list

    Returns:
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT
    """
    gQT = []
    T_gQT = []
    len_idx = []
    for I in I_list:
        gQT.append(hapi.TIPS_2017_ISOQ_HASH[(M, I)])
        T_gQT.append(hapi.TIPS_2017_ISOT_HASH[(M, I)])
        len_idx.append(len(hapi.TIPS_2017_ISOQ_HASH[(M, I)]))

    # pad gQT and T_gQT with the last element
    len_max = np.max(len_idx)
    for idx, iso in enumerate(I_list):
        l_add = [gQT[idx][-1]] * (len_max - len(gQT[idx]))
        gQT[idx] = np.append(gQT[idx], l_add)

        l_add = [T_gQT[idx][-1]] * (len_max - len(T_gQT[idx]))
        T_gQT[idx] = np.append(T_gQT[idx], l_add)

    return jnp.array(gQT), jnp.array(T_gQT)


def read_path(path):
    """HITRAN IO for a HITRAN/HITEMP par file.

    Args:
        path: HITRAN/HITEMP par file
    Returns:
        numinf: nu minimum for multiple file cases of HITEMP (H2O and CO2)
        numtag: tag for wavelength range

    Note:
       For H2O and CO2, HITEMP provides multiple par files. numinf and numtag are the ranges and identifiers for the multiple par files.
    """
    warnings.warn("We recommend to use spec.api for HITRAN/HITEMP I/O",
                  DeprecationWarning)
    exception = False
    if "01_HITEMP" in path.stem:
        exception = True
        numinf = np.array([
            0., 50., 150., 250., 350., 500., 600., 700., 800., 900., 1000.,
            1150., 1300., 1500., 1750., 2000., 2250., 2500., 2750., 3000.,
            3250., 3500., 4150., 4500., 5000., 5500., 6000., 6500., 7000.,
            7500., 8000., 8500., 9000., 11000.
        ])
        maxnu = 30000.
        numtag = make_numtag(numinf, maxnu)
    if "02_HITEMP" in path.stem:
        exception = True
        numinf = np.array([
            0., 500., 625., 750., 1000., 1500., 2000., 2125., 2250., 2500.,
            3000., 3250., 3500., 3750., 4000., 4500., 5000., 5500., 6000.,
            6500.
        ])
        maxnu = 12785.
        numtag = make_numtag(numinf, maxnu)

    if exception == False:
        numinf = None
        numtag = ''

    return numinf, numtag


def make_numtag(numinf, maxnu):
    """making numtag from numinf.

    Args:
        numinf: nu minimum for multiple file cases of HITEMP (H2O and CO2)
        maxnu:  maximum nu

    Returns:
        numtag: tag for wavelength range
    """
    numtag = []
    for i in range(len(numinf) - 1):
        imin = '{:05}'.format(int(numinf[i]))
        imax = '{:05}'.format(int(numinf[i + 1]))
        numtag.append(imin + '-' + imax)

    imin = imax
    imax = '{:05}'.format(int(maxnu))
    numtag.append(imin + '-' + imax)

    return numtag


def extract_hitemp(parbz2, nurange, margin, tag):
    """extract .par between nurange[0] and nurange[-1]

    Args:
       parbz2: .par.bz2 HITRAN/HITEMP file (str)
       nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid
       margin: margin for nurange (cm-1)
       tag: tag for directory and output file

    Return:
       path of output file (pathlib)
    """
    import os
    import bz2
    import tqdm
    import pathlib
    warnings.warn("We recommend to use spec.api for HITRAN/HITEMP I/O",
                  DeprecationWarning)
    infilepath = pathlib.Path(parbz2)
    outdir = infilepath.parent / pathlib.Path(tag)
    os.makedirs(str(outdir), exist_ok=True)
    outpath = outdir / pathlib.Path(infilepath.stem)

    numin = nurange[0] - margin
    numax = nurange[-1] + margin
    alllines = bz2.BZ2File(str(infilepath), 'r')

    f = open(str(outpath), 'w')
    for line in tqdm.tqdm(alllines, desc='Extract HITEMP'):
        nu = float(line[3:15])
        if nu <= numax and nu >= numin:
            if b'\r\n' in line[-2:]:
                f.write(line[:-2].decode('utf-8') + '\n')
            else:
                f.write(line.decode('utf-8'))
    alllines.close()
    f.close()
    return outpath


if __name__ == '__main__':
    nurange = [4200.0, 4300.0]
    margin = 1.0
    tag = 'ext'
    extract_hitemp('~/exojax/data/CH4/06_HITEMP2020.par.bz2', nurange, margin,
                   tag)
