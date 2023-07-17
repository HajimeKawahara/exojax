import pytest
import jax.numpy as jnp
import numpy as np
from exojax.linalg.tridiag import solve_tridiag


def solve_semitridiag_naive(diag, lower_diag, upper_diag, vector):
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


def solve_vmap_semitridiag_naive(diag, lower_diag, upper_diag, vector):
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

    return delta[:, 0]


def test_solve_vmapsemitridiag():
    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax import vmap
    nwav = 3
    mat = jnp.array([[1., 8., 0., 0.], [5., 2., 9., 0.], [0., 6., 3., 10.],
                     [0., 0., 7., 4.]])
    ans = jnp.array([4., 3., 2., 1.])
    vector = mat @ ans

    vvector = jnp.array([vector for _ in range(nwav)])
    diag = jnp.diag(mat)
    lower_diag = jnp.diag(mat, k=-1)
    upper_diag = jnp.diag(mat, k=1)
    vdiag = jnp.array([diag for _ in range(nwav)])
    vlower_diag = jnp.array([lower_diag for _ in range(nwav)])
    vupper_diag = jnp.array([upper_diag for _ in range(nwav)])

    x0 = solve_vmap_semitridiag_naive(vdiag, vlower_diag, vupper_diag, vvector)

    print(x0, "vmap tridiag")

    #for i in range(nwav):
    #    assert jnp.sum((mat @ x[i] - vector)**2) == 0.0


def test_solve_semitridiag():
    from jax.config import config
    config.update("jax_enable_x64", True)
    mat = jnp.array([[1., 8., 0., 0.], [5., 2., 9., 0.], [0., 6., 3., 10.],
                     [0., 0., 7., 4.]])
    ans = jnp.array([4., 3., 2., 1.])
    vector = mat @ ans
    diag = jnp.diag(mat)
    lower_diag = jnp.diag(mat, k=-1)
    upper_diag = jnp.diag(mat, k=1)

    x0 = solve_semitridiag_naive(diag, lower_diag, upper_diag, vector)
    print(x0)
    assert x0 == pytest.approx(ans[0])


def test_solve_tridiag():
    from jax.config import config
    config.update("jax_enable_x64", True)
    mat = jnp.array([[1., 8., 0., 0.], [5., 2., 9., 0.], [0., 6., 3., 10.],
                     [0., 0., 7., 4.]])
    ans = jnp.array([4., 3., 2., 1.])
    vector = mat @ ans
    diag = jnp.diag(mat)
    lower_diag = jnp.diag(mat, k=-1)
    upper_diag = jnp.diag(mat, k=1)

    x = solve_tridiag(diag, lower_diag, upper_diag, vector)
    print(x, "tridiag")

    assert jnp.sum((mat @ x - vector)**2) == 0.0


def test_solve_vmaptridiag():
    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax import vmap
    nwav = 3
    mat = jnp.array([[1., 8., 0., 0.], [5., 2., 9., 0.], [0., 6., 3., 10.],
                     [0., 0., 7., 4.]])
    ans = jnp.array([4., 3., 2., 1.])
    vector = mat @ ans

    vvector = jnp.array([vector for _ in range(nwav)])
    diag = jnp.diag(mat)
    lower_diag = jnp.diag(mat, k=-1)
    upper_diag = jnp.diag(mat, k=1)
    vdiag = jnp.array([diag for _ in range(nwav)])
    vlower_diag = jnp.array([lower_diag for _ in range(nwav)])
    vupper_diag = jnp.array([upper_diag for _ in range(nwav)])

    vsolve_tridiag = vmap(solve_tridiag, (0, 0, 0, 0), 0)

    x = vsolve_tridiag(vdiag, vlower_diag, vupper_diag, vvector)

    print(x, "vmap tridiag")

    for i in range(nwav):
        assert jnp.sum((mat @ x[i] - vector)**2) == 0.0


def test_solve_tridiag_for_the_same_length_diags_if_ignoring_the_last_elements(
):
    from jax.config import config
    config.update("jax_enable_x64", True)
    mat = jnp.array([[1., 8., 0., 0.], [5., 2., 9., 0.], [0., 6., 3., 10.],
                     [0., 0., 7., 4.]])
    ans = jnp.array([4., 3., 2., 1.])
    vector = mat @ ans
    diag = jnp.diag(mat)
    lower_diag = jnp.hstack([jnp.diag(mat, k=-1), [10.0]])
    upper_diag = jnp.hstack([jnp.diag(mat, k=1), [100.0]])

    x = solve_tridiag(diag, lower_diag, upper_diag, vector)
    print(x, "tridiag")

    assert jnp.sum((mat @ x - vector)**2) == 0.0


if __name__ == "__main__":
    #test_solve_tridiag_for_the_same_length_diags_if_ignoring_the_last_elements()
    #test_solve_tridiag()
    test_solve_vmaptridiag()
    test_solve_vmapsemitridiag()
    #test_solve_tridiag()
    #test_solve_semitridiag()