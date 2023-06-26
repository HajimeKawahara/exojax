import jax.numpy as jnp


def compute_tridiag_diagonals(scat_coeff, trans_coeff, upper_diagonal_top,
                              diagonal_top, diagonal_btm, lower_diagonal_btm):
    """computes the diagonals from scattering and transmission coefficients for the tridiagonal system

    Args:
        scat_coeff (_type_): scattering coefficient of the layer, S_n
        trans_coeff (_type_): transmission coefficient of the layer T_n
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

    Sn_minus_one = jnp.roll(scat_coeff, 1)  #S_{n-1}
    Tn_minus_one = jnp.roll(trans_coeff, 1)  #T_{n-1}
    Sn_plus_one = jnp.roll(scat_coeff, -1)  # S_{n+1}

    diagonal = scat_coeff * (Sn_minus_one**2 - Tn_minus_one**2)
    upper_diagonal = Sn_minus_one * trans_coeff

    lower_diagonal = Sn_plus_one * trans_coeff

    #boundary setting
    upper_diagonal = upper_diagonal.at[0].set(upper_diagonal_top)
    diagonal = diagonal.at[0].set(diagonal_top)
    diagonal = diagonal.at[-1].set(diagonal_btm)
    lower_diagonal = lower_diagonal.at[-2].set(lower_diagonal_btm)

    return diagonal, lower_diagonal, upper_diagonal


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
    delta = zeta_plus**2 - zeta_minus**2
    theta = zeta_plus * zeta_minus
    tn = lambdan * dtau
    eplus = jnp.exp(tn)
    eminus = jnp.exp(-tn)
    denom = zeta_plus**2 * eplus - zeta_minus**2 * eminus
    trans_coeff = delta / denom
    scat_coeff = theta * (eplus - eminus) / denom
    return trans_coeff, scat_coeff


def toon_zetalambda_coeffs(gamma_1, gamma_2):
    """computes coupling coefficients zeta and lambda coefficients for Toon-type two stream approximation 

    Args:
        gamma_1 (_type_): Toon+89 gamma_1 coefficient
        gamma_2 (_type_): Toon+89 gamma_2 coefficient

    Returns:
        _type_: coupling zeta (+), coupling zeta (-), lambda coefficients
    """
    delta = jnp.sqrt((gamma_1 - gamma_2) / (gamma_1 + gamma_2))
    zeta_plus = 0.5 * (1.0 + delta)
    zeta_minus = 0.5 * (1.0 - delta)
    lambdan = jnp.sqrt(gamma_1**2 - gamma_2**2)
    return zeta_plus, zeta_minus, lambdan


def sh2_zetalambda_coeff():
    raise ValueError("not implemented yet.")


def test_tridiag_coefficients():
    import numpy as np
    Nlayer = 4
    S = jnp.array(range(0, Nlayer)) + 1
    T = (jnp.array(range(0, Nlayer)) + 1) * 2
    boundaries = (1.0, 2.0, 3.0, 4.0)
    upper_diagonal_top, diagonal_top, diagonal_btm, lower_diagonal_btm = boundaries

    diag, lower, upper = compute_tridiag_diagonals(S, T, upper_diagonal_top,
                                                   diagonal_top, diagonal_btm,
                                                   lower_diagonal_btm)

    ref_diag = jnp.array([
        diagonal_top, S[1] * (S[0]**2 - T[0]**2), S[2] * (S[1]**2 - T[1]**2),
        diagonal_btm
    ])
    assert np.all(ref_diag - diag) == 0.0
    ref_lower = jnp.array([S[1] * T[0], S[2] * T[1], lower_diagonal_btm])
    assert np.all(ref_lower - lower[:-1]) == 0.0
    ref_upper = jnp.array([upper_diagonal_top, S[0] * T[1], S[1] * T[2]])
    assert np.all(ref_upper - upper[:-1]) == 0.0


if __name__ == "__main__":
    test_tridiag_coefficients()
