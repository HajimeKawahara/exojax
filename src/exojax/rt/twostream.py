""" Two-stream solvers and related methods 

    Note:
        ExoJAX has two types of the flux-based two-stream solvers for scattering/reflection. 
        - fluxadding 
        - LART

"""

import jax.numpy as jnp
from jax.lax import scan


def solve_fluxadding_twostream(
    trans_coeff, scat_coeff, reduced_source_function, reflectivity_bottom, source_bottom
):
    """Two-stream RT solver using flux adding

    Args:
        trans_coeff (_type_): Transmission coefficient
        scat_coeff (_type_): Scattering coefficient
        reduced_source_function :  pi \mathcal{B} (Nlayer, Nnus)
        reflectivity_bottom (_type_): R^+_N (Nnus)
        source_bottom (_type_): S^+_N (Nnus)

    Returns:
        Effective reflectivity (hat(R^plus)), Effective source (hat(S^plus))
    """

    Rplus, Splus = compute_fluxadding_coeffs(
        trans_coeff,
        scat_coeff,
        reduced_source_function,
        reflectivity_bottom,
        source_bottom,
    )
    return Rplus[0], Splus[0]


def compute_fluxadding_coeffs(
    trans_coeff, scat_coeff, reduced_source_function, reflectivity_bottom, source_bottom
):
    """Computes upward effective reflection/source at all interfaces.

    Args:
        trans_coeff: Transmission coefficient (Nlayer, Nnus).
        scat_coeff: Scattering coefficient (Nlayer, Nnus).
        reduced_source_function: Reduced source function (Nlayer, Nnus).
        reflectivity_bottom: Bottom boundary reflectivity (Nnus).
        source_bottom: Bottom boundary source (Nnus).

    Returns:
        tuple: Rplus, Splus, each with shape (Nlayer + 1, Nnus).
    """

    pihatB = (1.0 - trans_coeff - scat_coeff) * reduced_source_function

    # bottom reflection
    Rplus_bottom = reflectivity_bottom
    Splus_bottom = source_bottom

    def f(carry_ip1, arr):
        Rplus_prev, Splus_prev = carry_ip1
        scat_coeff_i, trans_coeff_i, pihatB_i = arr
        denom = 1.0 - scat_coeff_i * Rplus_prev
        Splus_each = (
            pihatB_i + trans_coeff_i * (Splus_prev + pihatB_i * Rplus_prev) / denom
        )
        Rplus_each = scat_coeff_i + trans_coeff_i**2 * Rplus_prev / denom
        RS = [Rplus_each, Splus_each]
        return RS, RS

    # main loop
    arrin = [
        scat_coeff[::-1],
        trans_coeff[::-1],
        pihatB[::-1],
    ]
    _, stackedRS = scan(f, [Rplus_bottom, Splus_bottom], arrin)
    Rplus_reverse, Splus_reverse = stackedRS

    Rplus = jnp.vstack([Rplus_reverse[::-1], Rplus_bottom])
    Splus = jnp.vstack([Splus_reverse[::-1], Splus_bottom])
    return Rplus, Splus


def compute_fluxadding_downward_coeffs(
    trans_coeff, scat_coeff, reduced_source_function, source_top=None
):
    """Computes downward effective reflection/source at all interfaces.

    The top boundary is expressed as:
    F_0^- = 0 * F_0^+ + source_top.

    Args:
        trans_coeff: Transmission coefficient (Nlayer, Nnus).
        scat_coeff: Scattering coefficient (Nlayer, Nnus).
        reduced_source_function: Reduced source function (Nlayer, Nnus).
        source_top: Top boundary incoming flux (Nnus). Defaults to zero.

    Returns:
        tuple: Rminus, Sminus, each with shape (Nlayer + 1, Nnus).
    """

    _, Nnus = trans_coeff.shape
    if source_top is None:
        source_top = jnp.zeros(Nnus)

    pihatB = (1.0 - trans_coeff - scat_coeff) * reduced_source_function
    Rminus_top = jnp.zeros(Nnus)
    Sminus_top = source_top

    def f(carry_i, arr):
        Rminus_prev, Sminus_prev = carry_i
        scat_coeff_i, trans_coeff_i, pihatB_i = arr
        denom = 1.0 - scat_coeff_i * Rminus_prev
        Sminus_each = (
            pihatB_i + trans_coeff_i * (Sminus_prev + pihatB_i * Rminus_prev) / denom
        )
        Rminus_each = scat_coeff_i + trans_coeff_i**2 * Rminus_prev / denom
        RS = [Rminus_each, Sminus_each]
        return RS, RS

    arrin = [scat_coeff, trans_coeff, pihatB]
    _, stackedRS = scan(f, [Rminus_top, Sminus_top], arrin)
    Rminus_layers, Sminus_layers = stackedRS

    Rminus = jnp.vstack([Rminus_top, Rminus_layers])
    Sminus = jnp.vstack([Sminus_top, Sminus_layers])
    return Rminus, Sminus


def solve_fluxadding_twostream_fluxes(
    trans_coeff,
    scat_coeff,
    reduced_source_function,
    reflectivity_bottom,
    source_bottom,
    source_top=None,
):
    """Computes two-stream upward and downward fluxes at all interfaces.

    Args:
        trans_coeff: Transmission coefficient (Nlayer, Nnus).
        scat_coeff: Scattering coefficient (Nlayer, Nnus).
        reduced_source_function: Reduced source function (Nlayer, Nnus).
        reflectivity_bottom: Bottom boundary reflectivity (Nnus).
        source_bottom: Bottom boundary source (Nnus).
        source_top: Top boundary incoming flux (Nnus). Defaults to zero.

    Returns:
        tuple: flux_plus, flux_minus, each with shape (Nlayer + 1, Nnus).
    """

    Rplus, Splus = compute_fluxadding_coeffs(
        trans_coeff,
        scat_coeff,
        reduced_source_function,
        reflectivity_bottom,
        source_bottom,
    )
    Rminus, Sminus = compute_fluxadding_downward_coeffs(
        trans_coeff, scat_coeff, reduced_source_function, source_top
    )

    denom = 1.0 - Rplus * Rminus
    flux_plus = (Rplus * Sminus + Splus) / denom
    flux_minus = Rminus * flux_plus + Sminus
    return flux_plus, flux_minus


def solve_lart_twostream(diagonal, lower_diagonal, upper_diagonal, vector, flux_bottom):
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

    # top boundary
    That0 = upper_diagonal[0, :] / diagonal[0, :]
    Qhat0 = vector[0, :] / diagonal[0, :]

    # main loop
    arrin = [
        diagonal[1:nlayer, :],
        lower_diagonal[0 : nlayer - 1, :],
        upper_diagonal[1:nlayer, :],
        vector[1:nlayer, :],
    ]
    _, stackedTQ = scan(f, [That0, Qhat0], arrin)
    That, Qhat = stackedTQ

    # inserts top boundary
    That = jnp.insert(jnp.array(That), 0, That0, axis=0)
    Qhat = jnp.insert(jnp.array(Qhat), 0, Qhat0, axis=0)

    # (no)surface term
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
    trans_func = jnp.exp(-lambdan * dtau)  # transmission function (Heng 2017, 3.58)
    denom = zeta_plus**2 - (zeta_minus * trans_func) ** 2
    trans_coeff = trans_func * (zeta_plus**2 - zeta_minus**2) / denom
    scat_coeff = (1.0 - trans_func**2) * zeta_plus * zeta_minus / denom

    return trans_coeff, scat_coeff


def compute_tridiag_diagonals_and_vector(
    scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top, vector_top
):
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
    vector = rn_minus * hatpiB - rn * (Tn_minus_one - Sn_minus_one) * hatpiB_minus_one

    # top bundary
    vector = vector.at[0].set(vector_top)

    return diagonal, lower_diagonal, upper_diagonal, vector
