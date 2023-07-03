import jax.numpy as jnp

def set_scat_trans_coeffs(zeta_plus, zeta_minus, lambdan, dtau):
    """sets scattering and transmission coefficients from zeta and lambda coefficient and dtau

    Args:
        zeta_plus (_type_): coupling zeta (+) coefficient (e.g. Heng 2017)
        zeta_minus (_type_): coupling zeta (-) coefficient (e.g. Heng 2017)
        lambdan (_type_): lambda coefficient
        dtau (_type_): optical depth interval of the layers

    Returns:
        _type_: transmission coefficient, scattering coeffcient
    """
    trans_func = jnp.exp(-lambdan * dtau) # transmission function (Heng 2017, 3.58)
    denom = zeta_plus**2 - (zeta_minus * trans_func)**2
    trans_coeff = trans_func * (zeta_plus**2 - zeta_minus**2) / denom
    scat_coeff = (1.0 - trans_func**2) * zeta_plus * zeta_minus / denom
    return trans_coeff, scat_coeff


def compute_tridiag_diagonals(scat_coeff, trans_coeff, upper_diagonal_top,
                              diagonal_top, diagonal_btm, lower_diagonal_btm):
    """computes the diagonals from scattering and transmission coefficients for the tridiagonal system

    Args:
        scat_coeff (_type_): scattering coefficient of the n-th layer, S_n
        trans_coeff (_type_): transmission coefficient of the n-th layer, T_n
        upper_diagonal_top (_type_): a[0] upper diagonal top boundary 
        diagonal_top (_type_): b[0] diagonal top boundary
        diagonal_btm (_type_): b[N-1] diagonal bottom boundary
        lower_diagonal_btm (_type_): c[N-2] lower diagonal bottom boundary

    Notes:
        While diagonal (b_n) has the Nlayer-dimension, upper and lower diagonals (an and cn) should have Nlayer-1 dimension originally.
        However, the tridiagonal solver linalg.tridiag.solve_tridiag ignores the last elements of upper and lower diagonals.
        Therefore, we leave the last elements of the upper and lower diagonals as is. Do not use these elements.

    Returns:
        _jnp arrays: diagonal [Nlayer], lower dianoals [Nlayer], upper diagonal [Nlayer], 
    """

    Sn_minus_one = jnp.roll(scat_coeff, 1)  # S_{n-1}
    Tn_minus_one = jnp.roll(trans_coeff, 1)  # T_{n-1}
    Sn_plus_one = jnp.roll(scat_coeff, -1)  # S_{n+1}

    upper_diagonal = Sn_minus_one * trans_coeff  # an
    diagonal = scat_coeff * \
        (Sn_minus_one**2 - Tn_minus_one**2) - Sn_minus_one  # bn
    lower_diagonal = Sn_plus_one * trans_coeff  # cn

    # boundary setting
    upper_diagonal = upper_diagonal.at[0].set(upper_diagonal_top)
    diagonal = diagonal.at[0].set(diagonal_top)
    diagonal = diagonal.at[-1].set(diagonal_btm)
    lower_diagonal = lower_diagonal.at[-2].set(lower_diagonal_btm)

    return diagonal, lower_diagonal, upper_diagonal




def sh2_zetalambda_coeff():
    raise ValueError("not implemented yet.")



#if __name__ == "__main__":
#   test_tridiag_coefficients()
