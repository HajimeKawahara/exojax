"""functions for computation of line shape density (LSD) 

   * there are both numpy and jnp versions. (np)*** is numpy version.
   * (np)getix provides the contribution and index.
   * (np)add(x)D constructs the (x)Dimensional LSD array given the contribution and index.
   * uniqidx does not have the contribution.
   * uniqidx(_2D) provides the indices based on the unique values (vectors) of the input array. These are the numpy version.

"""
import numpy as np
from jax.numpy import index_exp as joi
import jax.numpy as jnp
from jax import jit, vmap
from exojax.utils.constants import hcperk, Tref
import tqdm

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

def npgetix_exp(x, xv, Ttyp):
    """numpy version of getix weigthed by exp(-hc/kT).

    Args:
        x: x array
        xv: x grid
        Ttyp: typical temperature for the temperature correction

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """

    if Ttyp is not None:
        x=np.exp(-hcperk*x*(1.0/Ttyp-1.0/Tref))
        xv=np.exp(-hcperk*xv*(1.0/Ttyp-1.0/Tref))
    
    indarr = np.arange(len(xv))
    pos = np.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index    

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
    a = a.at[joi[ix, iy]].add(w*(1-cx)*(1-cy))
    a = a.at[joi[ix, iy+1]].add(w*(1-cx)*cy)
    a = a.at[joi[ix+1, iy]].add(w*cx*(1-cy))
    a = a.at[joi[ix+1, iy+1]].add(w*cx*cy)
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
    a = a.at[joi[ix, iy, iz]].add(w*(1-cx)*(1-cy)*(1-cz))
    a = a.at[joi[ix, iy+1, iz]].add(w*(1-cx)*cy*(1-cz))
    a = a.at[joi[ix+1, iy, iz]].add(w*cx*(1-cy)*(1-cz))
    a = a.at[joi[ix+1, iy+1, iz]].add(w*cx*cy*(1-cz))
    a = a.at[joi[ix, iy, iz+1]].add(w*(1-cx)*(1-cy)*cz)
    a = a.at[joi[ix, iy+1, iz+1]].add(w*(1-cx)*cy*cz)
    a = a.at[joi[ix+1, iy, iz+1]].add(w*cx*(1-cy)*cz)
    a = a.at[joi[ix+1, iy+1, iz+1]].add(w*cx*cy*cz)
    return a

def npadd1D(a, w, cx, ix):
    """numpy version: Add into an array when contirbutions and indices are given (1D).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 

    Returns:
        a

    """
    np.add.at(a, ix, w*(1-cx))
    np.add.at(a, ix+1, w*cx)
    return a

def npadd2D(a, w, cx, ix, cy, iy):
    """numpy version: Add into an array when contirbutions and indices are given (2D).

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
    np.add.at(a, (ix, iy), w*(1-cx)*(1-cy))
    np.add.at(a, (ix+1, iy), w*cx*(1-cy))
    np.add.at(a, (ix+1, iy+1), w*cx*cy)
    np.add.at(a, (ix, iy+1), w*(1-cx)*cy)
    return a

def npadd3D_uniqidx(a, w, cx, ix, cy, iy, uiz):
    """numpy version: Add into an array when contirbutions and indices are given (3D=2D+uniqidx).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        cy: given contribution for y 
        iy: given index for y
        uiz: given unique index for z

    Returns:
        a(N, ny, nx )

    """
    np.add.at(a, (ix, uiz, iy), w*(1-cx)*(1-cy))
    np.add.at(a, (ix+1, uiz, iy), w*cx*(1-cy))
    np.add.at(a, (ix+1, uiz, iy+1), w*cx*cy)
    np.add.at(a, (ix, uiz, iy+1), w*(1-cx)*cy)

    
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

def uniqidx(a):
    """ compute indices based on uniq values of the input array.

    Args:
        a: input array

    Returns:
        unique index, unique value
    
    Examples:
        
        >>> a=np.array([4,7,7,7,8,4])
        >>> uidx, uval=uniqidx(a) #-> [0,1,1,1,2,0], [4,7,8]

    
    """
    uniqvals=np.unique(a)
    uidx=np.where(a==uniqvals[0], 0, None)
    for i,uv in enumerate(uniqvals[1:]):
        uidx=np.where(a==uv, i+1, uidx)
    return uidx, uniqvals

def uniqidx_2D(a):
    """ compute indices based on uniq values of the input array (2D).

    Args:
        a: input array (N,M), will use unique M-dim vectors

    Returns:
        unique index, unique value

    Examples:
        
        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])
        >>> uidx, uval=uniqidx_2D(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]

    """
    N,_=np.shape(a)
    uniqvals=np.unique(a,axis=0)
    uidx=np.zeros(N,dtype=int)
    uidx_p=np.where(a==uniqvals[0], True, False)
    uidx[np.array(np.prod(uidx_p,axis=1),dtype=bool)]=0
    for i,uv in enumerate(tqdm.tqdm(uniqvals[1:],desc="uniqidx")):
        uidx_p=np.where(a==uv, True, False)
        uidx[np.array(np.prod(uidx_p,axis=1),dtype=bool)]=i+1
    return uidx, uniqvals
