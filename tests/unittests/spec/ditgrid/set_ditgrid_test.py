import jax.numpy as jnp
import pytest
import numpy as np
from exojax.spec.set_ditgrid import ditgrid_log_interval, ditgrid_linear_interval


def test_ditgrid_log_interval():
    x = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
    val = ditgrid_log_interval(x, dit_grid_resolution=0.1, adopt=True)
    diff = np.log(val[1:]) - np.log(val[:-1])
    ref = np.ones(len(diff)) * 0.09594105
    assert np.all(diff == pytest.approx(ref))


def test_ditgrid_linear_interval():
    x = np.array([0.5, 0.3, 0.4, 0.47])
    Tfix = 3000.0
    Tref = 296.0
    weight = np.log(Tfix) - np.log(Tref)
    val = ditgrid_linear_interval(x,
                                  dit_grid_resolution=0.1,
                                  weight=weight,
                                  adopt=True)
    diff = weight * val[1:] - weight * val[:-1]
    ref = np.ones(len(diff)) * 0.09264032
    assert np.all(diff == pytest.approx(ref))


if __name__ == "__main__":
    #    test_ditgrid_log_interval()
    test_ditgrid_linear_interval()
