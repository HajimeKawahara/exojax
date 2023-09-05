from exojax.spec.twostream import compute_tridiag_diagonals_and_vector
from exojax.spec.twostream import solve_lart_twostream_numpy
from exojax.spec.twostream import solve_lart_twostream

import jax.numpy as jnp
import numpy as np


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


def test_solve_lart_twostream_numpy():
    """
    This test does not guarantee that solve_lart_twostream_numpy itself but checks the results are consistent with the fiducial ones
    """
    _, _, upper_diagonal_top, diagonal_top, vector_top, piB, scat_coeff, trans_coeff = samples_lart_flux2st(
    )
    diagonal, lower_diagonal, upper_diagonal, vector = compute_tridiag_diagonals_and_vector(
        scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top,
        vector_top)

    cumTtilde, Qtilde, spectrum = solve_lart_twostream_numpy(
        diagonal, lower_diagonal, upper_diagonal, vector)
    ref = 0.0072969906
    res = (spectrum[0] - ref)**2
    assert res < 1.e-15

def test_solve_lart_twostream_by_comparing_with_numpy_version():
    _, _, upper_diagonal_top, diagonal_top, vector_top, piB, scat_coeff, trans_coeff = samples_lart_flux2st(
    )
    diagonal, lower_diagonal, upper_diagonal, vector = compute_tridiag_diagonals_and_vector(
        scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top,
        vector_top)
    cumTtilde_np, Qtilde_np, spectrum_np = solve_lart_twostream_numpy(
        diagonal, lower_diagonal, upper_diagonal, vector)
    cumTtilde, Qtilde, spectrum = solve_lart_twostream(
        diagonal, lower_diagonal, upper_diagonal, vector)
    assert np.array_equal(Qtilde, Qtilde_np)

if __name__ == "__main__":
    #test_scat_lart_flux2st_tridiag_coefficients()
    test_solve_lart_twostream_numpy()
    test_solve_lart_twostream_by_comparing_with_numpy_version()