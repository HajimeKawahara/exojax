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

        
def compute_tridiag_diagonals_and_vector(scat_coeff, trans_coeff, piBplus, upper_diagonal_top, diagonal_top, vector_top):
    """computes the diagonals and right-handside vector from scattering and transmission coefficients for the tridiagonal system

    Args:
        scat_coeff (_type_): scattering coefficient of the n-th layer, S_n
        trans_coeff (_type_): transmission coefficient of the n-th layer, T_n
        piBplus (): upward Planck source function, piB^+_n 
        upper_diagonal_top (_type_): a[0] upper diagonal top boundary 
        diagonal_top (_type_): b[0] diagonal top boundary
        vector_top (_type_): vector top boundary 
    Notes:
        While diagonal (b_n) has the Nlayer-dimension, upper and lower diagonals (an and cn) should have Nlayer-1 dimension originally.
        However, the tridiagonal solver linalg.tridiag.solve_tridiag ignores the last elements of upper and lower diagonals.
        Therefore, we leave the last elements of the upper and lower diagonals as is. Do not use these elements.

    Returns:
        _jnp arrays: diagonal [Nlayer], lower dianoals [Nlayer], upper diagonal [Nlayer], vector [Nlayer], 
    """ 

    Sn_minus_one = jnp.roll(scat_coeff, 1, axis=0)  # S_{n-1}
    Tn_minus_one = jnp.roll(trans_coeff, 1, axis=0)  # T_{n-1}
    Sn_plus_one = jnp.roll(scat_coeff, -1, axis=0)  # S_{n+1}

    upper_diagonal = Sn_minus_one  # an
    diagonal = scat_coeff * (Tn_minus_one**2 - Sn_minus_one**2) + Sn_minus_one  # bn
    diagonal = diagonal/trans_coeff
    lower_diagonal = Sn_plus_one  # cn

    # top boundary setting
    upper_diagonal = upper_diagonal.at[0].set(upper_diagonal_top) 
    diagonal = diagonal.at[0].set(diagonal_top) 

    # vector
    piBplus_plus_one = jnp.roll(piBplus, -1, axis=0)  # piBlus_{n+1}
    cpiBplus_minus_one = jnp.roll(lower_diagonal*piBplus, 1, axis=0)  # c_{n-1}*piBplus_{n-1}
    vector = - upper_diagonal*piBplus_plus_one + diagonal*piBplus - cpiBplus_minus_one

    # top bundary
    vector = vector.at[0].set(vector_top)

    return diagonal, lower_diagonal, upper_diagonal, vector


def sh2_zetalambda_coeff():
    raise ValueError("not implemented yet.")

def rtrun_emis_toon(dtau, source_matrix):
    """Radiative Transfer using the Toon-type two-stream approximaion 

    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    zeta_plus, zeta_minus, lambdan = toon_zetalambda_coeffs(gamma_1, gamma_2)
    set_scat_trans_coeffs(zeta_plus, zeta_minus, lambdan, dtau)



#if __name__ == "__main__":
#   test_tridiag_coefficients()
