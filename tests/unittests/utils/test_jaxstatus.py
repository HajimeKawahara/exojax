import pytest
from jax import config

from exojax.utils.jaxstatus import check_jax64bit


def test_check_raise_valueerror_when_32bit():
    config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="JAX 32bit mode is not allowed"):
        check_jax64bit(allow_32bit=False)
