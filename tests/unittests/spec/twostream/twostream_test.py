from exojax.spec.twostream import compute_tridiag_diagonals
import jax.numpy as jnp


def test_tridiag_coefficients():
    import numpy as np
    Nlayer = 4
    S = jnp.array(range(0, Nlayer)) + 1.
    T = (jnp.array(range(0, Nlayer)) + 1) * 2.
    boundaries = (1.0, 2.0, 3.0, 4.0)
    upper_diagonal_top, diagonal_top, diagonal_btm, lower_diagonal_btm = boundaries

    diag, lower, upper = compute_tridiag_diagonals(S, T, upper_diagonal_top,
                                                   diagonal_top, diagonal_btm,
                                                   lower_diagonal_btm)

    ref_diag = jnp.array([
        diagonal_top, S[1] * (S[0]**2 - T[0]**2) -S[0] , S[2] * (S[1]**2 - T[1]**2) -S[1],
        diagonal_btm
    ])
    assert np.array_equal(ref_diag,diag) 
    ref_lower = jnp.array([S[1] * T[0], S[2] * T[1], lower_diagonal_btm])
    assert np.array_equal(ref_lower, lower[:-1])
    ref_upper = jnp.array([upper_diagonal_top, S[0] * T[1], S[1] * T[2]])
    assert np.array_equal(ref_upper, upper[:-1])


if __name__ == "__main__":
    test_tridiag_coefficients()
