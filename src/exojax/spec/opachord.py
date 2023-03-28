import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def chord_geometric_matrix(height, radius_lower):
    """compute chord geometric matrix

    Args:
        height (1D array): (normalized) height of the layers from top atmosphere, Nlayer
        radius_lower (1D array): (normalized) radius at the lower boundary from top to bottom (R0), (Nlayer)

    Returns:
        2D array: chord geometric matrix (Nlayer, Nlayer), lower triangle matrix

    Notes:
        Our definitions of the radius_lower and height (and radius_top, internally defined) are as follows:
        n=0,1,...,N-1
        radius_lower[N-1] = radius_btm (i.e. R0)    
        radius_lower[n-1] = radius_lower[n] + height[n]
        radius_top = radius_lower[0] + height[0]
        
    """
    radius_top = radius_lower[0] + height[0]  #radius at TOA
    radius_shifted = jnp.roll(radius_lower, 1)  # r_{k-1}
    radius_shifted = radius_shifted.at[0].set(radius_top)
    fac_right = jnp.sqrt(radius_lower[None, :]**2 - radius_lower[:, None]**2)
    fac_left = jnp.sqrt(radius_shifted[None, :]**2 - radius_lower[:, None]**2)
    raw_matrix = 2.0 * (fac_left - fac_right) / height
    return jnp.tril(raw_matrix)


def chord_optical_depth(chord_geometric_matrix, dtau):
    """chord optical depth vector from a chord geometric matrix and dtau
    
    Args:
        chord_geometric_matrix (jnp array): chord geometric matrix (Nlayer, Nlayer), lower triangle matrix 
        dtau (jnp array): layer optical depth matrix, dtau (Nlayer, N_wavenumber)

    Returns: chord optical depth (tauchord) matrix (Nlayer, N_wavenumber)

    """
    return jnp.dot(chord_geometric_matrix, dtau)
