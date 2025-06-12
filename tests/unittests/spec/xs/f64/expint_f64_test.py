import numpy as np
from scipy.special import expn
from exojax.rt import rtransfer as rt
import jax.numpy as jnp
import pytest
from jax import config                                                 #
config.update("jax_enable_x64", True)


def test_comparison_expint():

    x=np.logspace(-4,1.9,1000)
    dif=2.0*expn(3,x)-rt.trans2E3(x)
    assert np.max(dif) < 4.e-8

if __name__ == "__main__":
    test_comparison_expint()

