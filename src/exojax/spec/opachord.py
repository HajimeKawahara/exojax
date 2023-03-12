import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def chord_geometric_matrix(height, radius, radius_btm):
    """compute chord geometric matrix

    Args:
        height (1D array): (normalized) height of the layers from top atmosphere, Nlayer
        radius (1D array): (normalized) radius of the layers from top atmosphere, Nlayer
        radius_btm (float): radius at the bottom of the layeres. If using normalized radius, 1.0. 
    Returns:
        2D array: chord geometric matrix (Nlayer, Nlayer)
    """
    radius_shifted = jnp.roll(radius, -1) #r_{k+1}
    radius_shifted = radius_shifted.at[-1].set(radius_btm)
    


    #height = height.at[0].set(jnp.inf)

    fac_left = jnp.sqrt(radius[None, :]**2 - radius_shifted[:, None]**2)
    fac_right = jnp.sqrt(radius_shifted[None, :]**2 - radius_shifted[:, None]**2)
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
