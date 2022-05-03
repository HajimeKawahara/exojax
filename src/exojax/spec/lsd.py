"""functions to compute line shape matrices

"""
from jax.numpy import index_exp as joi
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

def getix(x, xv):
    """jnp version of getix.

    Args:
        x: x array
        xv: x grid, should be ascending order 

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.

    Example:

       >>> from exojax.spec.lsd import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))
    """
    indarr = jnp.arange(len(xv))
    pos = jnp.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index


def npgetix(x, xv):
    """numpy version of getix.

    Args:
        x: x array
        xv: x grid, should be ascending order

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """
    indarr = np.arange(len(xv))
    pos = np.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index


@jit
def inc3D_givenx(a, w, cx, ix, y, z, xv, yv, zv):
    """Compute integrated neighbouring contribution for the 3D lineshape distribution (LSD) matrix (memory reduced sum) but using given contribution and index for x .

    Args:
        a: lineshape density (LSD) array (jnp.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        y: y values (N)
        z: z values (N)
        xv: x grid
        yv: y grid
        zv: z grid            

    Returns:
        lineshape distribution matrix (integrated neighbouring contribution for 3D)

    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n \otimes fz_n, 
        where w_n is the weight, fx_n, fy_n, and fz_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 

    """

    cy, iy = getix(y, yv)
    cz, iz = getix(z, zv)

    a = a.at[joi[ix, iy, iz]].add(w*(1-cx)*(1-cy)*(1-cz))
    a = a.at[joi[ix, iy+1, iz]].add(w*(1-cx)*cy*(1-cz))
    a = a.at[joi[ix+1, iy, iz]].add(w*cx*(1-cy)*(1-cz))
    a = a.at[joi[ix+1, iy+1, iz]].add(w*cx*cy*(1-cz))
    a = a.at[joi[ix, iy, iz+1]].add(w*(1-cx)*(1-cy)*cz)
    a = a.at[joi[ix, iy+1, iz+1]].add(w*(1-cx)*cy*cz)
    a = a.at[joi[ix+1, iy, iz+1]].add(w*cx*(1-cy)*cz)
    a = a.at[joi[ix+1, iy+1, iz+1]].add(w*cx*cy*cz)

    return a

@jit
def inc2D_givenx(a, w, cx, ix, y, yv):
    """Compute integrated neighbouring contribution for 2D LSD (memory reduced sum) but using given contribution and index for x .

    Args:
        a: lineshape density (LSD) array (jnp.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        y: y values (N)
        yv: y grid

    Returns:
        lineshape distribution matrix (integrated neighbouring contribution for 2D)

    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n, 
        where w_n is the weight, fx_n, fy_n,  are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 

    """

    cy, iy = getix(y, yv)

    a = a.at[joi[ix, iy]].add(w*(1-cx)*(1-cy))
    a = a.at[joi[ix, iy+1]].add(w*(1-cx)*cy)
    a = a.at[joi[ix+1, iy]].add(w*cx*(1-cy))
    a = a.at[joi[ix+1, iy+1]].add(w*cx*cy)

    return a

    
