import jax.numpy as jnp


def solve_twostream_lart_numpy(diagonal, lower_diagonal, upper_diagonal, vector):
    """Two-stream RT solver given tridiagonal system components (LART form) but numpy version

    Args:
        diagonal (_type_): diagonal component of the tridiagonal system (bn)
        lower_diagonal (_type_): lower diagonal component of the tridiagonal system (cn)
        upper_diagonal (_type_): upper diagonal component of the tridiagonal system (an)
        vector (_type_): right-hand side vector (dn)

    Note:
        Our definition of the tridiagonal components is 
        an F+_(n+1) + bn F+_n + c_(n-1) F+_(n-1) = dn 
        Notice that c_(n-1) is not cn

    Returns:
        _type_: cumlative T, tilde Q, spectrum 
    """
    import numpy as np
    nlayer, _ = diagonal.shape
    Ttilde = np.zeros_like(diagonal)
    Qtilde = np.zeros_like(diagonal)
    Ttilde[0, :] = upper_diagonal[0, :] / diagonal[0, :]
    Qtilde[0, :] = vector[0, :] / diagonal[0, :]
    for i in range(1, nlayer):
        gamma = diagonal[i, :] - lower_diagonal[i - 1, :] * Ttilde[i - 1, :]
        Ttilde[i, :] = upper_diagonal[i, :] / gamma
        Qtilde[i, :] = (vector[i, :] +
                        lower_diagonal[i - 1, :] * Qtilde[i - 1, :]) / gamma

    cumTtilde = np.cumprod(Ttilde, axis=0)
    contribution_function = cumTtilde * Qtilde
    spectrum = np.nansum(contribution_function, axis=0)
    return cumTtilde, Qtilde, spectrum

def solve_twostream_pure_absorption_numpy(trans_coeff, scat_coeff, piB):
    """solve pure absorption limit for two stream

    Args:
        trans_coeff (_type_): _description_
        scat_coeff (_type_): _description_
        piB (_type_): _description_

    Returns:
        _type_: cumlative T, tilde Q, spectrum 
    """
    import numpy as np
    Qpure = np.zeros_like(trans_coeff)
    nlayer, nwav = trans_coeff.shape
    for i in range(0, nlayer - 1):
        Qpure[i, :] = (1.0 - trans_coeff[i, :] - scat_coeff[i, :]) * piB[i, :]
    cumTpure = np.cumprod(trans_coeff, axis=0)
    spectrum_pure = np.nansum(cumTpure * Qpure, axis=0)
    return cumTpure, Qpure, spectrum_pure


def contribution_function_lart(cumT, Q):
    return cumT * Q

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

        
def compute_tridiag_diagonals_and_vector(scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top, vector_top):
    """computes the diagonals and right-handside vector from scattering and transmission coefficients for the tridiagonal system

    Args:
        scat_coeff (_type_): scattering coefficient of the n-th layer, S_n
        trans_coeff (_type_): transmission coefficient of the n-th layer, T_n
        piB (): Planck source function, piB
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
    
    rn = scat_coeff/trans_coeff
    rn_plus_one = jnp.roll(rn, -1, axis=0)
    rn_minus = Sn_minus_one/trans_coeff

    # Case I 
    upper_diagonal = Sn_minus_one  # an
    diagonal = rn*(Tn_minus_one**2 - Sn_minus_one**2) + rn_minus # bn
    lower_diagonal = rn_plus_one*trans_coeff  # cn


    # top boundary setting
    upper_diagonal = upper_diagonal.at[0].set(upper_diagonal_top) 
    diagonal = diagonal.at[0].set(diagonal_top) 

    # vector
    hatpiB = (1.0 - trans_coeff - scat_coeff)*piB
    hatpiB_minus_one = jnp.roll(hatpiB, 1, axis=0)  
    vector = rn_minus*hatpiB - rn*(1.0 - Sn_minus_one)*hatpiB_minus_one

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
