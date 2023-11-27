"""interp test

"""
import numpy as np


def test_interp2d():
    xp = np.array([1.0, 4.0])
    yp = np.array([2.0, 4.0])
    fp = np.zeros((2, 2))
    fp[:, 0] = np.array([1.0, 3.0])
    fp[:, 1] = np.array([2.0, 4.0])

    f = interp2d(3.0, 3.0, xp, yp, fp)


from exojax.utils.indexing import getix


def interp2d(x, y, xp, yp, fp):
    cx, ix = getix(x, xp)
    cy, iy = getix(y, yp)
    ax = (fp[ix+1,iy] - fp[ix,iy])/(xp[ix+1]-xp[ix])
    ay = (fp[ix,iy+1] - fp[ix,iy])/(yp[iy+1]-yp[iy])
    return ax*cx + ay*cy + fp[ix,iy]  