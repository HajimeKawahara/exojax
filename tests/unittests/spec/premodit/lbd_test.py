"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from jax import grad
from exojax.utils.constants import hcperk
from exojax.spec.lbd import weight_point1, weight_point2
import warnings
from jax.config import config

config.update("jax_enable_x64", True)


def lbd_coefficients(elower_lines,
                     elower_grid,
                     Tref,
                     Twp,
                     conversion_dtype=np.float64):
    """

    Notes:
        This function is practically an extension of spec.lsd.npgetix_exp. 
        Twp (Temperature at the weight point) corresponds to Ttyp, but not typical temparture at all.  

    Args:
        x: x array
        xv: x grid
        Tref: reference tempreature to be used for the line strength S0
        Ttyp: typical temperature for the temperature correction
        converted_dtype: data type for conversion. Needs enough large because this code uses exp.
        
    Returns:
        the zeroth coefficient at Point 2
        the first coefficient at Point 2
        index

    Note:
       zeroth and first coefficients are at Point 2, i.e. for i=index+1. 
       1 - zeroth_coefficient gives the zeroth coefficient at Point 1, i.e. for i=index.
       - first_coefficient gives the first coefficient at Point 1.   
    """

    xl = np.array(elower_lines, dtype=conversion_dtype)
    xi = np.array(elower_grid, dtype=conversion_dtype)
    xl = np.exp(-hcperk * xl * (1.0 / Twp - 1.0 / Tref))
    xi = np.exp(-hcperk * xi * (1.0 / Twp - 1.0 / Tref))

    # check overflow
    check_overflow(conversion_dtype, xl, xi)

    zeroth_coeff, index = compute_contribution_and_index(xl, xi)
    index = index.astype(int)
    x1 = xi[index]
    x2 = xi[index + 1]
    E1 = elower_grid[index]
    E2 = elower_grid[index + 1]
    dx = x2 - x1
    #derivative of x
    xp1 = -hcperk * E1 * x1
    xp2 = -hcperk * E2 * x2
    xpl = -hcperk * elower_lines * xl

    first_coeff = ((xpl - xp1) * dx - (xl - x1) * (xp2 - xp1)) / dx**2

    return zeroth_coeff, first_coeff, index


def test_lbd_coefficients():
    """We check here consistency with lbderror.weight_point2_dE
    """
    from exojax.spec.lbderror import weight_point2_dE
    from jax import grad
    from jax.config import config
    config.update("jax_enable_x64", True)

    elower_lines = np.array([70.0, 130.0])
    elower_grid = np.array([0.0, 100.0, 200.0])
    Tref = 300.0
    Twp = 700.0
    p1 = 0.7
    p2 = 0.3
    dE = 100.0
    c0, c1, i = lbd_coefficients(elower_lines, elower_grid, Tref, Twp)
    cref0a = weight_point2_dE(1.0 / Twp, 1.0 / Tref, dE, p1)
    cref0b = weight_point2_dE(1.0 / Twp, 1.0 / Tref, dE, p2)
    cref1a = grad(weight_point2_dE, argnums=0)(1.0 / Twp, 1.0 / Tref, dE, p1)
    cref1b = grad(weight_point2_dE, argnums=0)(1.0 / Twp, 1.0 / Tref, dE, p2)

    assert c0[0] == pytest.approx(cref0a)
    assert c0[1] == pytest.approx(cref0b)
    assert c1[0] == pytest.approx(cref1a)
    assert c1[1] == pytest.approx(cref1b)

def check_overflow(conversion_dtype, x_, xv_):
    if np.isinf(np.max(x_)) or np.isinf(np.max(xv_)):
        print("\n conversion_dtype = ", conversion_dtype, "\n")
        raise ValueError("Use larger conversion_dtype.")


def compute_contribution_and_index(xl, xi):
    indarr = np.arange(len(xi))
    pos = np.interp(xl, xi, indarr)
    zeroth_coefficient, index = np.modf(pos)
    return zeroth_coefficient, index


def lbd_first_approx(T, Twp, lbd_zeroth_Twp, lbd_first_Twp):
    """construct lbd from zeroth and first terms in the first Taylor approximation

    Args:
        T (_type_): _description_
        Twp (_type_): _description_
        lbd_zeroth_Twp (_type_): _description_
        lbd_first_Twp (_type_): _description_

    Returns:
        _type_: LBD (line basis density at T)
    """
    return lbd_zeroth_Twp + lbd_first_Twp * (1.0 / T - 1.0 / Twp)


def test_construct_lbd():
    return


if __name__ == "__main__":
    test_lbd_coefficients()
