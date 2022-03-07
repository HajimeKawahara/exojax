"""test for erfcx.

- This test compares hjert with scipy.erfcx, see Appendix in Paper I.
"""

import pytest
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.special import erfcx as sc_erfcx
from exojax.special import erfcx
from exojax.special._special import erfcx_scan

N = 10000
xv = jnp.logspace(-5, 5, N)
xvc = np.logspace(-5, 5, N)
verfcx = vmap(erfcx)
verfcx_scan = vmap(erfcx_scan)
ref = sc_erfcx(xvc)


def test_comparison_erfcx_scipy():
    d = (verfcx(xv) - ref) / ref
    assert np.max(d) < 2.e-6


def test_comparison_erfcx_scan_scipy():
    d = (verfcx_scan(xv) - ref) / ref
    assert np.max(d) < 2.e-6
