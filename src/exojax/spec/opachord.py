import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def chord_geometric_matrix(height, radius):
    """compute chord geometric matrix

    Args:
        height (1D array): (normalized) height of the layers from top atmosphere, Nlayer
        radius (1D array): (normalized) radius of the layers from top atmosphere, Nlayer

    Returns:
        2D array: chord geometric matrix (Nlayer, Nlayer)
    """
    radius_roll = jnp.roll(radius, 1)

    # elements at the top layer to be zero
    radius_roll = radius_roll.at[0].set(radius[0])
    height = height.at[0].set(jnp.inf)

    fac_right = jnp.sqrt(radius[None, :]**2 - radius[:, None]**2)
    fac_left = jnp.sqrt(radius_roll[None, :]**2 - radius[:, None]**2)
    raw_matrix = 2.0*(fac_left - fac_right) / height
    return jnp.tril(raw_matrix)


def tauchord(chord_geometric_matrix, xsmatrix):
    """chord opacity vector from a chord geometric matrix and xsmatrix
    
    Args:
        chord_geometric_matrix (jnp array): chord geometric matrix (Nlayer, Nlayer), lower triangle matrix 
        xsmatrix (jnp array): cross section matrix (Nlayer, N_wavenumber)

    Returns: tauchord matrix (Nlayer, N_wavenumber)

    """
    return jnp.dot(chord_geometric_matrix, xsmatrix)
