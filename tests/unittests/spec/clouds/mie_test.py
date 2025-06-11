from exojax.special.lognormal import cubeweighted_pdf
from exojax.database.mie import auto_rgrid
from exojax.database.mie import cubeweighted_integral_checker
import numpy as np


def test_autogrid():
    """tests robust sigmag range for auto_rgrid. Currently 1.0001,4 is within 1 % for the default setting
    """
    lower_limit_sigmag = 1.0001
    upper_limit_sigmag = 4.0
    rg_um = 0.05  # 0.1um = 100nm
    cm2um = 1.0e4
    cm2nm = 1.0e7
    rg = rg_um / cm2um  # in cgs
    rg_nm = rg * cm2nm
    sigr = np.linspace(lower_limit_sigmag, upper_limit_sigmag, 100)
    for sigmag in sigr:
        rgrid = auto_rgrid(rg_nm, sigmag)
        check = cubeweighted_integral_checker(rgrid, rg_nm, sigmag, accuracy=1.0e-2)
        if not check:
            import matplotlib.pyplot as plt
            val = cubeweighted_pdf(rgrid, rg_nm, sigmag)
            plt.plot(rgrid, val, ".")
            plt.show()
        assert check


def test_cubeweighted_integral_checker():
    rg_um = 0.05  # 0.1um = 100nm
    sigmag = 2.0
    cm2um = 1.0e4
    cm2nm = 1.0e7
    rg = rg_um / cm2um  # in cgs
    rg_nm = rg * cm2nm
    rgrid_lower = 1.0
    rgrid_upper = 10000.0
    nrgrid = 1000
    rgrid = np.linspace(rgrid_lower, rgrid_upper, nrgrid)

    check = cubeweighted_integral_checker(rgrid, rg_nm, sigmag)

    assert check


if __name__ == "__main__":
    test_cubeweighted_integral_checker()
    test_autogrid()
