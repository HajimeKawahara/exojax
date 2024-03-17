"""interp test

"""
import numpy as np
from exojax.utils.interp import interp2d_bilinear
from jax import config

config.update("jax_enable_x64", False)


def test_interp2d():
    # sets grids
    xp = np.array([1.0, 4.0, 6.0])
    yp = np.array([2.0, 4.0])
    fp = np.zeros((3, 2))
    fp[:, 0] = np.array([1.0, 3.0, 5.0])
    fp[:, 1] = np.array([2.0, 4.0, 7.0])
    # test coordinates
    x = np.array([2.5])
    y = np.array([3.0])
    # evaluates the interpolated value
    f = interp2d_bilinear(x, y, xp, yp, fp.T)

    assert f[0] == 2.5


if __name__ == "__main__":
    test_interp2d()
