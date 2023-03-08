import jax.numpy as jnp
import numpy as np


def tauchord(chord_matrix, xsmatrix):
    """chord opacity vector from a chord matrix and xsmatrix

    Note:
        transposed upper triangle matrix is like this
        [[1 2 3]
        [4 5 0]
        [7 0 0]]
        can be obtained by jnp.triu(jnp.array(square_matrix[:, ::-1]))[:,::-1]

    Args:
        chord_matrix (jnp array): chord square matrix (Nlayer, Nlayer), which is converted to a transposed upper triangle matrix 
        xsmatrix (jnp array): cross section matrix (Nlayer, N_wavenumber)

    Returns: tauchord matrix (Nlayer, N_wavenumber)

    """
    return jnp.dot(jnp.triu(chord_matrix[:,::-1])[:,::-1], xsmatrix)


def test_check_parallel_Ax_tauchord():
    A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = jnp.array([[1, 2, 3], [4, 5, 6]]).T
    Atranstriu = jnp.array([[1,2,3],[4,5,0],[7,0,0]])
    n = []
    for k in range(2):
        n.append(jnp.dot(Atranstriu, x[:, k]))
    n = jnp.array(n).T

    m = tauchord(A,x)
    
    assert np.all(m == n)

if __name__ == "__main__":
    test_check_parallel_Ax()