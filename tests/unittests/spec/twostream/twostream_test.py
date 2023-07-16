from exojax.spec.twostream import compute_tridiag_diagonals_and_vector
import jax.numpy as jnp


def test_tridiag_coefficients():
    import numpy as np
    Nlayer = 4
    S = jnp.array(range(0, Nlayer)) + 1.
    T = (jnp.array(range(0, Nlayer)) + 1) * 2.
    boundaries = (1.0, 2.0)
    upper_diagonal_top, diagonal_top = boundaries

    diag, lower, upper = compute_tridiag_diagonals_and_vector(S, T, upper_diagonal_top,
                                                   diagonal_top)#, diagonal_btm,
                                                   #lower_diagonal_btm)
    ref_diag = jnp.array([
        diagonal_top, S[1] * (S[0]**2 - T[0]**2) -S[0] , S[2] * (S[1]**2 - T[1]**2) -S[1],
        S[3] * (S[2]**2 - T[2]**2) -S[2]
    ])
    assert np.array_equal(ref_diag,diag) 
    ref_lower = jnp.array([S[1] * T[0], S[2] * T[1], S[3] * T[2]])
    assert np.array_equal(ref_lower, lower[:-1])
    ref_upper = jnp.array([upper_diagonal_top, S[0] * T[1], S[1] * T[2]])
    assert np.array_equal(ref_upper, upper[:-1])

def test_tridiag_coefficients_parallel_input():
    import numpy as np
    Nlayer = 4
    S = jnp.array(range(0, Nlayer)) + 1.
    T = (jnp.array(range(0, Nlayer)) + 1) * 2.
    upper_diagonal_top = jnp.array([1.0,1.0,1.0])
    diagonal_top = jnp.array([2.0,2.0,2.0])

    #Nparallel = 3
    Sarr = jnp.array([S,S,S]).T
    Tarr = jnp.array([T,T,T]).T
    
    
    diag, lower, upper = compute_tridiag_diagonals_and_vector(Sarr, Tarr, upper_diagonal_top,
                                                   diagonal_top)
                                                   
    ref_diag = jnp.array([
        diagonal_top, S[1] * (S[0]**2 - T[0]**2) -S[0] , S[2] * (S[1]**2 - T[1]**2) -S[1],
        S[3] * (S[2]**2 - T[2]**2) -S[2]
    ])
    ref_diag_arr = jnp.array([ref_diag,ref_diag,ref_diag]).T
    
    assert np.array_equal(ref_diag_arr,diag) 

    print(diag)


if __name__ == "__main__":
    test_tridiag_coefficients()
    test_tridiag_coefficients_parallel_input()  