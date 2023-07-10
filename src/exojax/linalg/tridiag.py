""" Tridiagonal Solver 

- The original code of solve_tridiag was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

"""
import jax.numpy as jnp
from jax.lax import scan


def solve_tridiag(diagonal, lower_diagonal, upper_diagonal, vector):
    """Tridiagonal Linear Solver for A x = b, using Thomas Algorithm  
    
    the original code was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

    Args:
        diagonal (1D array): the diagonal component vector of a matrix A [N]
        lower_diagonal (1D array): the lower diagonal component vector of a matrix A, [N-1] or [N] but lower_diagonal[-1] is ignored
        upper_diagonal (1D array): the upper diagonal component vector of a matrix A, [N-1] or [N] but upper_diagonal[-1] is ignored
        vector (1D array): the vector b [N]

    Notes:
        notation from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        _p indicates prime, ie. `d_p` is the variable name for d' on wikipedia
        


    Returns:
        1D array: solution vector (x)
    """

    size = len(diagonal)
    print("SIZE=",size)
    def thomas_scan(prev_cd_carry, bd):
        c_p, d_p, step = prev_cd_carry
        # the index of `a` doesn't matter at step 0 as
        # we won't use it at all. Same for `c` at final step
        a_index = jnp.where(step > 0, step - 1, 0)
        c_index = jnp.where(step < size, step, 0)

        b, d = bd
        a, c = lower_diagonal[a_index], upper_diagonal[c_index]
        denom = b - a * c_p
        new_d_p = (d - a * d_p) / denom
        new_c_p = c / denom
        return (new_c_p, new_d_p, step + 1), (new_c_p, new_d_p)

    def backsub(prev_x_carry, cd_p):
        x_prev, step = prev_x_carry
        c_p, d_p = cd_p
        x_new = d_p - c_p * x_prev
        return (x_new, step + 1), x_new

    # not a dummy init! 0 is the proper value for all of these
    init_thomas = (0, 0, 0)
    init_backsub = (0, 0)
    diag_vec = (diagonal, vector)
    _, cd_p = scan(thomas_scan, init_thomas, diag_vec, unroll=32)
    _, solution = scan(backsub, init_backsub, cd_p, reverse=True, unroll=32)

    return solution



