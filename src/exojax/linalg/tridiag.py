""" Tridiagonal Solver 

- The original code of solve_tridiag was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

"""
import jax.numpy as jnp
from jax.lax import scan
import numpy as np

######## NEED REFACTORING!!! ######


def solve_semitridiag_naive_array(diag, lower_diag, upper_diag, vector):
    N = len(vector)

    beta = np.zeros(N)
    beta[N - 1] = diag[N - 1]
    for j in range(1, N):
        i = N - 1 - j
        beta[i] = diag[i] - upper_diag[i] * lower_diag[i] / beta[i + 1]

    delta = np.zeros(N)
    delta[N - 1] = vector[N - 1] / beta[N - 1]
    for j in range(1, N):
        i = N - 1 - j
        delta[i] = (vector[i] - upper_diag[i] * delta[i + 1]) / beta[i]

    return delta[0]


def solve_semitridiag_naive(diag, lower_diag, upper_diag, vector):
    N = len(vector)

    beta = diag[N - 1]
    delta = vector[N - 1] / beta
    for j in range(1, N):
        i = N - 1 - j
        beta = diag[i] - upper_diag[i] * lower_diag[i] / beta
        delta = (vector[i] - upper_diag[i] * delta) / beta

    return delta


def solve_vmap_semitridiag_naive_array(diag, lower_diag, upper_diag, vector):
    Nwav, N = vector.shape

    beta = np.zeros_like(vector)
    beta[:, N - 1] = diag[:, N - 1]
    for j in range(1, N):
        i = N - 1 - j
        beta[:,
             i] = diag[:,
                       i] - upper_diag[:, i] * lower_diag[:, i] / beta[:,
                                                                       i + 1]

    delta = np.zeros_like(vector)
    delta[:, N - 1] = vector[:, N - 1] / beta[:, N - 1]
    for j in range(1, N):
        i = N - 1 - j
        delta[:, i] = (vector[:, i] -
                       upper_diag[:, i] * delta[:, i + 1]) / beta[:, i]

    #return delta[:, 0]
    return beta, delta

def solve_vmap_semitridiag_naive(diag, lower_diag, upper_diag, vector):
    Nwav, N = vector.shape

    beta = diag[:, N - 1]
    delta = vector[:, N - 1] / beta

    import matplotlib.pyplot as plt
    fig = plt.figure()
    for j in range(1, N):
        i = N - 1 - j
        beta = np.array(diag[:, i] - upper_diag[:, i] * lower_diag[:, i] / beta)
        delta = np.array((vector[:, i] - upper_diag[:, i] * delta) / beta)
        if i == 69 or i == 68:
            plt.plot(beta[10300:10650], label=str(i) + " beta")
            #plt.plot(delta[10300:10650] / np.max(delta[10300:10650]),
            #         label=str(i) + " delta")
        #if j > 25 and j < 35:
        #    plt.plot(delta[10300:10650], label=str(i))
        #print("beta",i, beta[10300:10700])
        #print("delta",i, delta[10300:10700])
    plt.legend()
    plt.show()
    return delta


#################


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
    print("SIZE=", size)

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
