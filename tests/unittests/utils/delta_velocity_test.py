import jax.numpy as jnp
import numpy as np
from exojax.utils.delta_velocity import delta_velocity_from_resolution
from exojax.utils.delta_velocity import dvgrid_rigid_rotation


def test_delta_velocity_from_resolution():
    from exojax.utils.constants import c
    N = 60
    Rarray = np.logspace(2, 7, N)
    dv_np = c * np.log1p(1.0 / Rarray)
    dv = delta_velocity_from_resolution(Rarray)
    resmax = np.max(np.abs(dv / dv_np) - 1)
    assert resmax < 3. * 1.e-7


def test_minimum_dv_grid():
    resolution = 10**5
    vsini = 150.0  #km/s
    x = dvgrid_rigid_rotation(resolution, vsini)
    assert x[0] <= -1.0 and x[-1] >= 1.0
    assert x[1] >= -1.0 and x[-2] <= 1.0


if __name__ == "__main__":
    test_delta_velocity_from_resolution()
    test_minimum_dv_grid()