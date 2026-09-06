"""test for partfn_Fe.

- Test polynomial expansion of the partition function of iron by Irwin1981
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exojax.database.core_atom.pf import partfn_Fe


def test_partfn_Fe():
    tabulated = 5.62138956e0  # Table 1 of Irwin (1981)
    diff = np.log(partfn_Fe(16000)) - tabulated
    assert diff < 1e-8


@pytest.mark.parametrize("enable_x64,rtol", [(False, 3.0e-6), (True, 1.0e-11)])
def test_partfn_fe_matches_original_polynomial(enable_x64, rtol):
    jax.config.update("jax_enable_x64", enable_x64)
    temperatures = np.array([296.0, 1000.0, 3000.0, 6000.0, 10000.0, 16000.0])
    # Original Irwin coefficients in ascending powers of log(T).
    coefficients = [-1.15609527e3, 7.46597652e2, -1.92865672e2,
                    2.49658410e1, -1.61934455, 4.21182087e-2]
    expected = np.exp(sum(
        coefficient * np.log(temperatures) ** power
        for power, coefficient in enumerate(coefficients)
    ))
    dtype = np.float64 if enable_x64 else np.float32
    actual = jax.jit(partfn_Fe)(jnp.asarray(temperatures, dtype=dtype))

    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0)
