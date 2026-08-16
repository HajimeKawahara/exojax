import numpy as np
from scipy.special import expn
from exojax.rt import rtransfer as rt


def test_comparison_expint():
    from jax import config

    config.update("jax_enable_x64", True)

    x = np.logspace(-4, 1.9, 1000)
    dif = np.abs(2.0 * expn(3, x) - rt.trans2E3(x))
    assert np.max(dif) < 4.0e-8


if __name__ == "__main__":
    test_comparison_expint()
