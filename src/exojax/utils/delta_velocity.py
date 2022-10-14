import jax.numpy as jnp
from exojax.utils.constants import c

def delta_velocity_from_resolution(resolution):
    """delta velocity from spectral resolution R

    Args:
        resolution : spectral resolution

    Note: 
        See also [#294](https://github.com/HajimeKawahara/exojax/issues/294) and exojax/tests/figures/functions/delta_velocity_comp.py

    Returns:
        delta velocity
    """
    return c*jnp.log1p(1.0/resolution)