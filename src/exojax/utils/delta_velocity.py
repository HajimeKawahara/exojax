import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import c


def dvgrid_rigid_rotation(resolution, vsini_max):
    """generate delta velocity grid for a rigid rotation

    Args:
        resolution: spectral resolution
        vsini: maximum Vsini allowed (km/s)

    Returns:
        1D array: delta velocity grid 
    """
    dv = delta_velocity_from_resolution(resolution)
    Nk = (vsini_max / dv) + 1
    Nk = Nk.astype(int)
    return dv * np.arange(-Nk, Nk + 1)


def delta_velocity_from_resolution(resolution):
    """delta velocity from spectral resolution R

    Args:
        resolution : spectral resolution

    Note: 
        See also [#294](https://github.com/HajimeKawahara/exojax/issues/294) and exojax/tests/figures/functions/delta_velocity_comp.py

    Returns:
        delta velocity
    """
    return c * jnp.log1p(1.0 / resolution)
