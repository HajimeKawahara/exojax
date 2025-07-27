"""test for hjert.

- This test compares hjert with scipy wofz, see Appendix in Paper I.
"""

from exojax.opacity.lpf.lpf import hjert
from scipy.special import wofz as sc_wofz
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


def test_comparison_hjert_scipy():

    Na = 300
    vl = -3
    vm = 5
    xarrv = jnp.logspace(vl, vm, Na)
    xarr = xarrv[:, None]*jnp.ones((Na, Na))
    aarrv = jnp.logspace(vl, vm, Na)
    aarr = aarrv[None, :]*jnp.ones((Na, Na))

    # scipy
    def H(a, x):
        z = x+(1j)*a
        w = sc_wofz(z)
        return w.real

    # hjert
    def vhjert(a):
        return vmap(hjert, (0, None), 0)(xarrv, a)

    vvhjert = jit(vmap(vhjert, 0, 0))
    diffarr = (vvhjert(aarrv).T-H(aarr, xarr))/H(aarr, xarr)

    assert np.max(diffarr) < 1.e-6


if __name__ == '__main__':
    test_comparison_hjert_scipy()
