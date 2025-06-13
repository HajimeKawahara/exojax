"""test for ljert.

- This test compares ljert with scipy wofz
"""

import pytest
from exojax.opacity.lpf.lpf import ljert
from scipy.special import wofz as sc_wofz
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


def test_comparison_ljert_scipy():

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
        return w.imag

    # ljert
    def vljert(a):
        return vmap(ljert, (0, None), 0)(xarrv, a)

    vvljert = jit(vmap(vljert, 0, 0))
    diffarr = (vvljert(aarrv).T-H(aarr, xarr))/H(aarr, xarr)

    assert np.max(diffarr) < 7.e-5


if __name__ == '__main__':
    test_comparison_ljert_scipy()
