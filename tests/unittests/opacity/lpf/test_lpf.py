"""Compare Voigt profile components with SciPy over eight decades."""

from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import wofz

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from exojax.opacity.lpf.lpf import hjert, ljert


@pytest.mark.parametrize("use_x64", [False, True])
@pytest.mark.parametrize(
    "profile,component,tolerance",
    [(hjert, "real", 1.0e-6), (ljert, "imag", 7.0e-5)],
    ids=["hjert", "ljert"],
)
def test_voigt_profile_matches_scipy(use_x64, profile, component, tolerance):
    with enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        x = jnp.logspace(-3, 5, 300, dtype=dtype)
        a = jnp.logspace(-3, 5, 300, dtype=dtype)
        z = np.asarray(x, dtype=np.float64)[:, None] + 1j * np.asarray(
            a, dtype=np.float64
        )[None, :]
        expected = getattr(wofz(z), component)
        actual = jit(vmap(vmap(profile, (0, None)), (None, 0)))(x, a).T
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0.0)
