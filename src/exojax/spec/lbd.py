import numpy as np
from exojax.utils.constants import hcperk
from exojax.spec.lsd import npgetix

def lbd_coefficients(elower_lines,
                     elower_grid,
                     Tref,
                     Twt,
                     conversion_dtype=np.float64):
    """compute the LBD zeroth and first coefficients
    
    Args:
        elower_lines: Line's Elower 
        elower_grid: Elower grid for LBD
        Tref: reference tempreature to be used for the line strength S0
        Twt: temperature used for the weight coefficient computation 
        converted_dtype: data type for conversion. Needs enough large because this code uses exp.
        
    Returns:
        the zeroth coefficient at Point 2, equivalent to cont in spec.lsd.npgetix_exp
        the first coefficient at Point 2
        index

    Note:
        This function is practically an extension of spec.lsd.npgetix_exp. 
        Twp (Temperature at the weight point) corresponds to Ttyp, but not typical temparture at all.  
        zeroth and first coefficients are at Point 2, i.e. for i=index+1. 
        1 - zeroth_coefficient gives the zeroth coefficient at Point 1, i.e. for i=index.
        - first_coefficient gives the first coefficient at Point 1.   
    """

    xl = np.array(elower_lines, dtype=conversion_dtype)
    xi = np.array(elower_grid, dtype=conversion_dtype)
    xl = np.exp(-hcperk * xl * (1.0 / Twt - 1.0 / Tref))
    xi = np.exp(-hcperk * xi * (1.0 / Twt - 1.0 / Tref))

    _check_overflow(conversion_dtype, xl, xi)

    zeroth_coeff, index = npgetix(xl, xi)
    index = index.astype(int)
    x1 = xi[index]
    x2 = xi[index + 1]
    E1 = elower_grid[index]
    E2 = elower_grid[index + 1]
    dx = x2 - x1

    #derivative of x/(-c2)
    xp1 = E1 * x1
    xp2 = E2 * x2
    xpl = elower_lines * xl

    first_coeff = -hcperk * ((xpl - xp1) * dx - (xl - x1) *
                             (xp2 - xp1)) / dx**2
    return zeroth_coeff, first_coeff, index

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

