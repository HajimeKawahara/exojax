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
