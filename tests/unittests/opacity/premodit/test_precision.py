import numpy as np
import pytest
from jax import config

from exojax.opacity import OpaPremodit


def test_premodit_rejects_32bit_before_reading_database():
    config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="JAX 32bit mode is not allowed"):
        OpaPremodit(mdb=None, nu_grid=np.array([1000.0, 1001.0]))
