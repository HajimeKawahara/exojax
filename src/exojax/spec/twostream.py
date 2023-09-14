""" Two-stream solvers and related methods 

"""

import jax.numpy as jnp
from jax.lax import scan


def solve_lart_twostream_numpy(diagonal, lower_diagonal, upper_diagonal,
                               vector):
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
        _type_: cumlative T, hat Q, spectrum 
    """
    import numpy as np
    nlayer, Nnus = diagonal.shape

    nlayer, _ = diagonal.shape
    That = np.zeros_like(diagonal)
    Qhat = np.zeros_like(diagonal)
    That[0, :] = upper_diagonal[0, :] / diagonal[0, :]
    Qhat[0, :] = vector[0, :] / diagonal[0, :]

    for i in range(1, nlayer):  #nlayer - 1 ...
        gamma = diagonal[i, :] - lower_diagonal[i - 1, :] * That[i - 1, :]
        That[i, :] = upper_diagonal[i, :] / gamma
        Qhat[i, :] = (vector[i, :] +
                      lower_diagonal[i - 1, :] * Qhat[i - 1, :]) / gamma

    #(no)surface term
    Qhat = jnp.vstack([Qhat, np.zeros(Nnus)])
    cumThat = jnp.cumprod(jnp.vstack([jnp.ones(Nnus), That]), axis=0)
    contribution_function = cumThat * Qhat
    spectrum = np.nansum(contribution_function, axis=0)

    return cumThat, Qhat, spectrum


def solve_lart_twostream(diagonal, lower_diagonal, upper_diagonal, vector,
                         flux_bottom):
    """Two-stream RT solver given tridiagonal system components (LART form)

    Args:
        diagonal (_type_): diagonal component of the tridiagonal system (bn)
        lower_diagonal (_type_): lower diagonal component of the tridiagonal system (cn)
        upper_diagonal (_type_): upper diagonal component of the tridiagonal system (an)
        vector (_type_): right-hand side vector (dn)
        flux_bottom: bottom flux FB
        
    Note:
        Our definition of the tridiagonal components is 
        an F+_(n+1) + bn F+_n + c_(n-1) F+_(n-1) = dn 
        Notice that c_(n-1) is not cn

    Returns:
        _type_: cumlative hat{T}, hat{Q}, spectrum 
    """
    nlayer, Nnus = diagonal.shape

    # arguments of the scanning function f:
    # carry_i_1 = [That_{i-1}, Qhat_{i-1}]
    # arr = [diagonal[1:nlayer], lower_diagonal[0:nlayer-1], upper_diagonal[1:nlayer], vector[1,nlayer]]

    def f(carry_i_1, arr):
        That_i_1, Qhat_i_1 = carry_i_1
        diagonal_i, lower_diagonal_i_1, upper_diagonal_i, vector_i = arr
        gamma = diagonal_i - lower_diagonal_i_1 * That_i_1
        That_each = upper_diagonal_i / gamma
        Qhat_each = (vector_i + lower_diagonal_i_1 * Qhat_i_1) / gamma
        TQ = [That_each, Qhat_each]
        return TQ, TQ

    #top boundary
    That0 = upper_diagonal[0, :] / diagonal[0, :]
    Qhat0 = vector[0, :] / diagonal[0, :]

    #main loop
    arrin = [
        diagonal[1:nlayer, :], lower_diagonal[0:nlayer - 1, :],
        upper_diagonal[1:nlayer, :], vector[1:nlayer, :]
    ]
    _, stackedTQ = scan(f, [That0, Qhat0], arrin)
    That, Qhat = stackedTQ

    #inserts top boundary
    That = jnp.insert(jnp.array(That), 0, That0, axis=0)
    Qhat = jnp.insert(jnp.array(Qhat), 0, Qhat0, axis=0)

    #(no)surface term
    Qhat = jnp.vstack([Qhat, flux_bottom])
    cumThat = jnp.cumprod(jnp.vstack([jnp.ones(Nnus), That]), axis=0)
    spectrum = jnp.nansum(cumThat * Qhat, axis=0)

    return cumThat, Qhat, spectrum


def solve_twostream_pure_absorption_numpy(trans_coeff, scat_coeff, piB):
    """solves pure absorption limit for two stream

    Args:
        trans_coeff (_type_): transmission coefficient 
        scat_coeff (_type_):  scattering coefficient
        piB (_type_): pi x Planck function

    Returns:
        _type_: cumlative transmission, generalized source, spectrum 
    """
    import numpy as np
    Qpure = np.zeros_like(trans_coeff)
    nlayer, Nnus = trans_coeff.shape
    for i in range(0, nlayer - 1):
        Qpure[i, :] = (1.0 - trans_coeff[i, :] - scat_coeff[i, :]) * piB[i, :]

    Qpure = np.vstack([Qpure, np.zeros(Nnus)])
    cumTpure = np.cumprod(np.vstack([np.ones(Nnus), trans_coeff]), axis=0)
    spectrum_pure = np.nansum(cumTpure * Qpure, axis=0)
    return cumTpure, Qpure, spectrum_pure


def contribution_function_lart(cumT, Q):
    """computes the contribution function from LART cumlative transmission and generalized source 

    Args:
        cumT (_type_): cumlative transmission
        Q (_type_): generalized source

    Returns:
        _type_: contribution fnction in a vector form
    """
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
    trans_func = jnp.exp(-lambdan *
                         dtau)  # transmission function (Heng 2017, 3.58)
    denom = zeta_plus**2 - (zeta_minus * trans_func)**2
    trans_coeff = trans_func * (zeta_plus**2 - zeta_minus**2) / denom
    scat_coeff = (1.0 - trans_func**2) * zeta_plus * zeta_minus / denom
    return trans_coeff, scat_coeff


def compute_tridiag_diagonals_and_vector(scat_coeff, trans_coeff, piB,
                                         upper_diagonal_top, diagonal_top,
                                         vector_top):
    """computes the diagonals and right-handside vector from scattering and transmission coefficients for the tridiagonal system

    Args:
        scat_coeff (_type_): scattering coefficient of the n-th layer, S_n
        trans_coeff (_type_): transmission coefficient of the n-th layer, T_n
        piB (): Planck source function, piB
        upper_diagonal_top (_type_): a[0] upper diagonal top boundary 
        diagonal_top (_type_): b[0] diagonal top boundary
        vector_top (_type_): vector top boundary 

    Notes:
        In ExoJAX 2 paper, we assume the tridiagonal form as -an F_{n+1}^+ + b_n F_n^+ - cn F_{n-1}^+ = dn
    Returns:
        jnp arrays: diagonal (bn) [Nlayer], lower dianoals (cn) [Nlayer], upper diagonal (an) [Nlayer], vector (dn) [Nlayer], 
    """

    Sn_minus_one = jnp.roll(scat_coeff, 1, axis=0)  # S_{n-1}
    Tn_minus_one = jnp.roll(trans_coeff, 1, axis=0)  # T_{n-1}

    rn = scat_coeff / trans_coeff
    rn_plus_one = jnp.roll(rn, -1, axis=0)
    rn_minus = Sn_minus_one / trans_coeff

    # Case I
    upper_diagonal = Sn_minus_one  # an
    diagonal = rn * (Tn_minus_one**2 - Sn_minus_one**2) + rn_minus  # bn
    lower_diagonal = rn_plus_one * trans_coeff  # cn

    # top boundary setting
    upper_diagonal = upper_diagonal.at[0].set(upper_diagonal_top)
    diagonal = diagonal.at[0].set(diagonal_top)

    # vector
    hatpiB = (1.0 - trans_coeff - scat_coeff) * piB
    hatpiB_minus_one = jnp.roll(hatpiB, 1, axis=0)
    vector = rn_minus * hatpiB - rn * (Tn_minus_one -
                                       Sn_minus_one) * hatpiB_minus_one

    # top bundary
    vector = vector.at[0].set(vector_top)

    return diagonal, lower_diagonal, upper_diagonal, vector


def sh2_zetalambda_coeff():
    raise ValueError("not implemented yet.")


#if __name__ == "__main__":
#   test_tridiag_coefficients()
