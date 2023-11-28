"""multi d interpolation
"""

from exojax.utils.indexing import getix


def interp2d_bilinear(x, y, xp, yp, fp):
    """2D bilinear interpolation

    Args:
        x (float ir 1D array): x (or x array) you want know the value
        y (float or 1D array): y (or y array) you want know the value
        xp (1D array): x grid (x.size = M)
        yp (1D array): y grid (y.size = N)
        fp (2D or nD array): value grid (shape = (M,N) or (M,N,...))

    Returns:
        float or nD array: bilinear interpolated value(s) at (x,y)
    """
    cx, ix = getix(x, xp)
    cy, iy = getix(y, yp)
    val = (
        (1.0 - cx) * (1.0 - cy) * fp[ix, iy]
        + cx * (1.0 - cy) * fp[ix + 1, iy]
        + (1.0 - cx) * cy * fp[ix, iy + 1]
        + cx * cy * fp[ix + 1, iy + 1]
    )
    return val
