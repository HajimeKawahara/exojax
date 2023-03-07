"""functions for computation of line shape density (LSD) 

   * there are both numpy and jnp versions. (np)*** is numpy version.
   * (np)getix provides the contribution and index.
   * (np)add(x)D constructs the (x)Dimensional LSD array given the contribution and index.

"""
import numpy as np
from jax.numpy import index_exp
import jax.numpy as jnp
from jax import jit
from exojax.utils.progbar import print_progress

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
    cont, index = jnp.modf(pos)
    return cont, index.astype(int)


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
    cont, index = np.modf(pos)
    return cont, index.astype(int)


def add2D(a, w, cx, ix, cy, iy):
    """Add into an array when contirbutions and indices are given (2D).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        cy: given contribution for y 
        iy: given index for y

    Returns:
        a

    """
    a = a.at[index_exp[ix, iy]].add(w * (1 - cx) * (1 - cy))
    a = a.at[index_exp[ix, iy + 1]].add(w * (1 - cx) * cy)
    a = a.at[index_exp[ix + 1, iy]].add(w * cx * (1 - cy))
    a = a.at[index_exp[ix + 1, iy + 1]].add(w * cx * cy)
    return a


def add3D(a, w, cx, ix, cy, iy, cz, iz):
    """Add into an array when contirbutions and indices are given (3D).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        cy: given contribution for y 
        iy: given index for y
        cz: given contribution for z 
        iz: given index for z

    Returns:
        a

    """
    a = a.at[index_exp[ix, iy, iz]].add(w * (1 - cx) * (1 - cy) * (1 - cz))
    a = a.at[index_exp[ix, iy + 1, iz]].add(w * (1 - cx) * cy * (1 - cz))
    a = a.at[index_exp[ix + 1, iy, iz]].add(w * cx * (1 - cy) * (1 - cz))
    a = a.at[index_exp[ix + 1, iy + 1, iz]].add(w * cx * cy * (1 - cz))
    a = a.at[index_exp[ix, iy, iz + 1]].add(w * (1 - cx) * (1 - cy) * cz)
    a = a.at[index_exp[ix, iy + 1, iz + 1]].add(w * (1 - cx) * cy * cz)
    a = a.at[index_exp[ix + 1, iy, iz + 1]].add(w * cx * (1 - cy) * cz)
    a = a.at[index_exp[ix + 1, iy + 1, iz + 1]].add(w * cx * cy * cz)
    return a


def npadd3D_direct1D(a,
                     w,
                     cx,
                     ix,
                     direct_cy,
                     direct_iy,
                     cz,
                     iz,
                     sumx=1.0,
                     sumz=1.0):
    """numpy version: Add into an array when contirbutions and indices are given (2D+direct).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        direct_cy: direct contribution for y 
        direct_iy: direct index for y
        cz: given contribution for z
        iz: given index for z
        sumx: a sum of contribution for x at point 1 and point 2, default=1.0
        sumz: a sum of contribution for z at point 1 and point 2, default=1.0

    Returns:
        lineshape density a(nx,ny,nz)

    Note:
        sumx or sumz gives a sum of contribution at point 1 and point 2. 
        For the zeroth coeeficient, it should be 1.0
        while it should be 0.0 for the first coefficient.

    """

    conjugate_cx = sumx - cx
    conjugate_cz = sumz - cz

    np.add.at(a, (ix, direct_iy, iz),
              w * conjugate_cx * direct_cy * conjugate_cz)
    np.add.at(a, (ix + 1, direct_iy, iz), w * cx * direct_cy * conjugate_cz)
    np.add.at(a, (ix, direct_iy, iz + 1), w * conjugate_cx * direct_cy * cz)
    np.add.at(a, (ix + 1, direct_iy, iz + 1), w * cx * direct_cy * cz)
    return a


def npadd3D_multi_index(a,
                        w,
                        cx,
                        ix,
                        cz,
                        iz,
                        uidx,
                        multi_cont_lines,
                        neighbor_uidx,
                        sumx=1.0,
                        sumz=1.0):
    """ numpy version: Add into an array using multi_index system in y
    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        cz: given contribution for z
        iz: given index for z
        sumx: a sum of contribution for x at point 1 and point 2, default=1.0
        sumz: a sum of contribution for z at point 1 and point 2, default=1.0
    
    Returns:
        lineshape density a(nx,ny,nz)

    Note:
        sumx or sumz gives a sum of contribution at point 1 and point 2. 
        For the zeroth coeeficient, it should be 1.0
        while it should be 0.0 for the first coefficient.


    """
    conjugate_multi_cont_lines = 1.0 - multi_cont_lines

    print_progress(0, 4, "Making LSD:")
    # index position
    direct_iy = uidx
    direct_cy = np.prod(conjugate_multi_cont_lines, axis=1)
    a = npadd3D_direct1D(a, w, cx, ix, direct_cy, direct_iy, cz, iz)

    print_progress(1, 4, "Making LSD:")
    # index position + (1, 0)
    direct_iy = neighbor_uidx[uidx, 0]
    direct_cy = multi_cont_lines[:, 0] * conjugate_multi_cont_lines[:, 1]
    a = npadd3D_direct1D(a, w, cx, ix, direct_cy, direct_iy, cz, iz)

    print_progress(2, 4, "Making LSD:")
    # index position + (0, 1)
    direct_iy = neighbor_uidx[uidx, 1]
    direct_cy = conjugate_multi_cont_lines[:, 0] * multi_cont_lines[:, 1]
    a = npadd3D_direct1D(a, w, cx, ix, direct_cy, direct_iy, cz, iz)

    print_progress(3, 4, "Making LSD:")
    # index position + (1, 1)
    direct_iy = neighbor_uidx[uidx, 2]
    direct_cy = np.prod(multi_cont_lines, axis=1)
    a = npadd3D_direct1D(a, w, cx, ix, direct_cy, direct_iy, cz, iz)
    
    print_progress(4, 4, "Making LSD:")
    
    return a




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
    a = add3D(a, w, cx, ix, cy, iy, cz, iz)
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
    a = add2D(a, w, cx, ix, cy, iy)
    return a
