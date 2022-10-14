import jax.numpy as jnp
import numpy as np
from exojax.utils.delta_velocity import delta_velocity_from_resolution


def test_delta_velocity_from_resolution():
    from exojax.utils.constants import c
    N = 60
    Rarray = np.logspace(2, 7, N)
    dv_np = c*np.log1p(1.0 / Rarray)
    dv = delta_velocity_from_resolution(Rarray)
    resmax = np.max(np.abs(dv/dv_np)-1)
    assert resmax < 3.*1.e-7

if __name__ == "__main__":
    test_delta_velocity_from_resolution()
