import jax.numpy as jnp

from exojax.opacity.modit.modit import xsmatrix_zeroscan
from exojax.opacity.modit.modit import xsmatrix_vald


def test_xsmatrix_vald_zeroscan():
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0, 1003.0])
    cnuS = jnp.array([[0.5]])
    indexnuS = jnp.array([[1]])
    R = 1000.0
    pmarray = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])
    nsigmaDlS = jnp.array([[[0.1]]])
    ngammaLMS = jnp.array([[[0.2]]])
    SijMS = jnp.array([[[1.0e-20]]])
    dgm_ngammaLS = jnp.array([[[0.1, 1.0]]])

    actual = xsmatrix_vald(
        cnuS,
        indexnuS,
        R,
        pmarray,
        nsigmaDlS,
        ngammaLMS,
        SijMS,
        nu_grid,
        dgm_ngammaLS,
    )
    expected = xsmatrix_zeroscan(
        cnuS[0],
        indexnuS[0],
        R,
        pmarray,
        nsigmaDlS[0],
        ngammaLMS[0],
        SijMS[0],
        nu_grid,
        dgm_ngammaLS[0],
    )

    assert actual.shape == (1, 1, 4)
    assert jnp.array_equal(actual[0], jnp.abs(expected))
