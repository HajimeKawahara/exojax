"""Compare the scaled complementary error function with SciPy in both precisions."""

import jax.numpy as jnp
from jax import vmap
import numpy as np
import pytest
from scipy.special import erfcx as scipy_erfcx

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from exojax.special import erfcx


@pytest.mark.parametrize("use_x64", [False, True])
def test_erfcx_matches_scipy(use_x64):
    with enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        x = jnp.logspace(-5, 5, 10000, dtype=dtype)
        expected = scipy_erfcx(np.asarray(x, dtype=np.float64))
        actual = vmap(erfcx)(x)
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=0.0)
