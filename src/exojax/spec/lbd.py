import numpy as np
from exojax.utils.constants import hcperk
from exojax.spec.lsd import npgetix


def lbd_coefficients(elower_lines,
                     elower_grid,
                     Tref,
                     Twt,
                     diffmode=2,
                     conversion_dtype=np.float64):
    """compute the LBD zeroth and first coefficients
    
    Args:
        elower_lines: Line's Elower 
        elower_grid: Elower grid for LBD
        Tref: reference tempreature to be used for the line strength S0
        Twt: temperature used for the weight coefficient computation 
        diffmode (int): i-th Taylor expansion is used for the weight, default is 2.
        converted_dtype: data type for conversion. Needs enough large because this code uses exp.
        
    Returns:
        the zeroth coefficient at Point 2
        the first coefficient at Point 2
        index

    Note:
        Twp (Temperature at the weight point) corresponds to Ttyp, but not typical temparture at all.  
        zeroth and first coefficients are at Point 2, i.e. for i=index+1. 
        1 - zeroth_coefficient gives the zeroth coefficient at Point 1, i.e. for i=index.
        - first_coefficient gives the first coefficient at Point 1.   
    """

    xl = np.array(elower_lines, dtype=conversion_dtype)
    xi = np.array(elower_grid, dtype=conversion_dtype)

    if Twt == Tref:
        print("Premodit: Twt = Tref = ", Twt, "K")
    else:
        print("Premodit: Twt=", Twt, "K Tref=", Tref, "K")
        xl = np.exp(-hcperk * xl * (1.0 / Twt - 1.0 / Tref))
        xi = np.exp(-hcperk * xi * (1.0 / Twt - 1.0 / Tref))

    _check_overflow(conversion_dtype, xl, xi)

    if Twt < Tref:
        xi = xi[::-1]

    zeroth_coeff, index = npgetix(xl, xi)
    index = index.astype(int)

    if Twt < Tref:
        zeroth_coeff = 1.0 - zeroth_coeff
        # zeroth_coeff = (xl - x1) / dx should give the same values
        index = len(xi) - index - 2
        xi = xi[::-1]

    if diffmode == 0:
        return [zeroth_coeff, None, None], index

    x1 = xi[index]
    x2 = xi[index + 1]
    E1 = elower_grid[index]
    E2 = elower_grid[index + 1]
    dx = x2 - x1

    # first derivative of x/(-c2)
    xp1 = E1 * x1
    xp2 = E2 * x2
    xpl = elower_lines * xl
    dxp = xp2 - xp1
    dr = xl - x1
    drp = xpl - xp1
    fac1 = (drp * dx - dr * dxp)
    first_coeff = -hcperk * fac1 / dx**2
    if diffmode == 1:
        return [zeroth_coeff, first_coeff, None], index

    # second derivative of (x/c2**2)
    dxpp = E2 * xp2 - E1 * xp1
    drpp = elower_lines * xpl - E1 * xp1
    #second_coeff = (drpp * dx + drp * dxp - drp * dxp - dr * dxpp) / dx**2 - fac1 * 2.0 * dx * dxp / dx**4
    second_coeff = drpp / dx - dr * dxpp / dx**2 - 2.0 * fac1  * dxp / dx**3
    second_coeff = hcperk**2 * second_coeff
    if diffmode == 2:
        return [zeroth_coeff, first_coeff, second_coeff], index
    
    raise ValueError("diffmode > 2 is not compatible yet.")

def _check_overflow(conversion_dtype, x_, xv_):
    if np.isinf(np.max(x_)) or np.isinf(np.max(xv_)):
        print("\n conversion_dtype = ", conversion_dtype, "\n")
        raise ValueError("Use larger conversion_dtype.")


def weight(T, Twt, zeroth_coeff, first_coeff):
    """construct lbd from zeroth and first terms in the first Taylor approximation

    Args:
        Tref: reference tempreature to be used for the line strength S0
        Twt: temperature used for the weight coefficient computation 
        
        T (float): _description_
        Twt (float): _description_
        zeroth_coeff (ndarray): zeroth coefficient of the Taylor expansion of the weight at Twt
        first_coeff (ndarray): first coefficient of the Taylor expansion of the weight at Twt

    Returns:
        ndarray: weight at the points 1 and 2
    """
    weight1 = (1.0 - zeroth_coeff) - first_coeff * (1.0 / T - 1.0 / Twt)
    weight2 = zeroth_coeff + first_coeff * (1.0 / T - 1.0 / Twt)

    return weight1, weight2
