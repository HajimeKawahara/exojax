from exojax.rt.twostream import compute_tridiag_diagonals_and_vector
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)


def samples_fluxadding_flux2st():
    Nlayer = 4
    B = jnp.array(range(0, Nlayer)) + 1.
    S = jnp.array(range(0, Nlayer)) + 1.
    T = (jnp.array(range(0, Nlayer)) + 1) * 2.
    piB = jnp.array([B, B, B]).T
    scat_coeff = jnp.array([S, S, S]).T*0.1
    trans_coeff = jnp.array([T, T, T]).T*0.1
    return piB, scat_coeff, trans_coeff




def samples_lart_flux2st():
    Nlayer = 4
    B = jnp.array(range(0, Nlayer)) + 1.
    S = jnp.array(range(0, Nlayer)) + 1.
    T = (jnp.array(range(0, Nlayer)) + 1) * 2.
    upper_diagonal_top = jnp.array([1.0, 1.0, 1.0])
    diagonal_top = jnp.array([2.0, 2.0, 2.0])
    vector_top = jnp.array([1.0, 1.0, 1.0])
    piB = jnp.array([B, B, B]).T
    scat_coeff = jnp.array([S, S, S]).T
    trans_coeff = jnp.array([T, T, T]).T
    return S, T, upper_diagonal_top, diagonal_top, vector_top, piB, scat_coeff, trans_coeff


def test_scat_lart_flux2st_tridiag_coefficients():
    """manual check of the scat LART tridiagonal coefficients
    """
    S, T, upper_diagonal_top, diagonal_top, vector_top, piB, scat_coeff, trans_coeff = samples_lart_flux2st(
    )

    diagonal, lower_diagonal, upper_diagonal, vector = compute_tridiag_diagonals_and_vector(
        scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top,
        vector_top)

    ref_diag = jnp.array([
        -T[0] * diagonal_top[0], S[1] * (S[0]**2 - T[0]**2) - S[0],
        S[2] * (S[1]**2 - T[1]**2) - S[1], S[3] * (S[2]**2 - T[2]**2) - S[2]
    ])
    ref_diag = -ref_diag / T
    ref_diag_arr = jnp.array([ref_diag, ref_diag, ref_diag]).T
    assert np.array_equal(ref_diag_arr, diagonal)


if __name__ == "__main__":
    
    test_scat_lart_flux2st_tridiag_coefficients()
    # test_solve_lart_twostream_numpy()
    # test_solve_lart_twostream_by_comparing_with_numpy_version()

