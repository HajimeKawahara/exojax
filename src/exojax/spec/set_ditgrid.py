"""set DIT 

* This module provides functions to generate the grid used in DIT/MODIT/PreMODIT.

"""

import numpy as np

def ditgrid(x, dit_grid_resolution=0.1, adopt=True):
    """DIT GRID.

    Args:
        x: simgaD or gammaL array (Nline)
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        grid for DIT
    """
    if np.min(x) <= 0.0:
        print('Warning: there exists negative or zero gamma. MODIT/DIT does not support this case.')

    lxmin = np.log(np.min(x))
    lxmax = np.log(np.max(x))
    lxmax = np.nextafter(lxmax, np.inf, dtype=lxmax.dtype)

    dlog = lxmax-lxmin
    Ng = int(dlog/dit_grid_resolution)+2
    if adopt == False:
        grid = np.exp(np.linspace(lxmin, lxmin+(Ng-1)*dit_grid_resolution, Ng))
    else:
        grid = np.exp(np.linspace(lxmin, lxmax, Ng))
    return grid

def ditgrid_matrix(x, res=0.1, adopt=True):
    """DIT GRID MATRIX.

    Args:
        x: simgaD or gammaL matrix (Nlayer x Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)
    """
    mmin = np.log(np.min(x, axis=1))
    mmax = np.log(np.max(x, axis=1))
    mmax = np.nextafter(mmax, np.inf, dtype=mmax.dtype)

    Nlayer = np.shape(mmax)[0]
    gm = []
    dlog = np.max(mmax-mmin)
    Ng = (dlog/res).astype(int)+2
    for i in range(0, Nlayer):
        lxmin = mmin[i]
        lxmax = mmax[i]
        if adopt == False:
            grid = np.exp(np.linspace(lxmin, lxmin+(Ng-1)*res, Ng))
        else:
            grid = np.exp(np.linspace(lxmin, lxmax, Ng))
        gm.append(grid)
    gm = np.array(gm)
    return gm


def minmax_ditgrid_matrix(x, dit_grid_resolution=0.1, adopt=True):
    """compute MIN and MAX DIT GRID MATRIX.

    Args:
        x: gammaL matrix (Nlayer x Nline)
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        minimum and maximum for DIT (dgm_minmax)
    """
    mmax = np.max(np.log10(x), axis=1)
    mmin = np.min(np.log10(x), axis=1)
    Nlayer = np.shape(mmax)[0]
    dgm_minmax = []
    dlog = np.max(mmax-mmin)
    Ng = (dlog/dit_grid_resolution).astype(int)+2
    for i in range(0, Nlayer):
        lxmin = mmin[i]
        lxmax = mmax[i]
        grid = [lxmin, lxmax]
        dgm_minmax.append(grid)
    dgm_minmax = np.array(dgm_minmax)
    return dgm_minmax

def precompute_modit_ditgrid_matrix(set_gm_minmax, dit_grid_resolution=0.1, adopt=True):
    """Precomputing MODIT GRID MATRIX for normalized GammaL.

    Args:
        set_gm_minmax: set of minmax of ditgrid matrix for different parameters [Nsample, Nlayers, 2], 2=min,max
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)
    """
    set_gm_minmax = np.array(set_gm_minmax)
    lminarray = np.min(set_gm_minmax[:, :, 0], axis=0)  # min
    lmaxarray = np.max(set_gm_minmax[:, :, 1], axis=0)  # max
    dlog = np.max(lmaxarray-lminarray)
    gm = []
    Ng = (dlog/dit_grid_resolution).astype(int)+2
    Nlayer = len(lminarray)
    for i in range(0, Nlayer):
        lxmin = lminarray[i]
        lxmax = lmaxarray[i]
        if adopt == False:
            grid = np.logspace(lxmin, lxmin+(Ng-1)*dit_grid_resolution, Ng)
        else:
            grid = np.logspace(lxmin, lxmax, Ng)
        gm.append(grid)
    gm = np.array(gm)
    return gm

