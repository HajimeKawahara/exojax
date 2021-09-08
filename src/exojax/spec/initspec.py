""" Initialization for opacity computation. The functions in this module are a wrapper for initialization processes for opacity computation.


"""
import jax.numpy as jnp
import numpy as np
from exojax.spec.dit import npgetix
from exojax.spec.make_numatrix import make_numatrix0

def init_lpf(nu_lines,nu_grid):
    """Initialization for LPF

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
       numatrix [Nline,Nnu]

    """
    numatrix=make_numatrix0(nu_grid,nu_lines,warning=True)
    return numatrix

def init_dit(nu_lines,nu_grid):
    """Initialization for DIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        cont (contribution) jnp.array
        index (index) jnp.array
        pmarray: (+1,-1) array whose length of len(nu_grid)+1

    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero.
    """
    if(nu_lines.dtype!=np.float64 and warning):
        print("Warning!: nu_lines is not np.float64 but ",nu.dtype)
    if(nu_grid.dtype!=np.float64 and warning):
        print("Warning!: nu_grid is not np.float64 but ",nu.dtype)

    
    cont,index=npgetix(nu_lines,nu_grid)
    dnu=nu_grid[1]-nu_grid[0]
    pmarray=np.ones(len(nu_grid)+1)
    pmarray[1::2]=pmarray[1::2]*-1

    return jnp.array(cont), jnp.array(index), pmarray

def init_modit(nu_lines,nu_grid):
    """Initialization for MODIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        cont: (contribution) jnp.array
        index: (index) jnp.array
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1


    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero. dq is computed using numpy not jnp.numpy. If you use jnp, you might observe a significant residual because of the float32 truncation error.


    """
    if(nu_lines.dtype!=np.float64 and warning):
        print("Warning!: nu_lines is not np.float64 but ",nu.dtype)
    if(nu_grid.dtype!=np.float64 and warning):
        print("Warning!: nu_grid is not np.float64 but ",nu.dtype)

    R=(len(nu_grid)-1)/np.log(nu_grid[-1]/nu_grid[0]) #resolution
    cont,index=npgetix(nu_lines,nu_grid)
    #dq=R*(np.log(nu_grid[1])-np.log(nu_grid[0]))
    pmarray=np.ones(len(nu_grid)+1)
    pmarray[1::2]=pmarray[1::2]*-1
    
    return jnp.array(cont), jnp.array(index), R, jnp.array(pmarray)

def init_redit(nu_lines,nu_grid):
    """Initialization for REDIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)

    Returns:
        cont (contribution) jnp.array
        index (index) jnp.array
        R: spectral resolution
        dq: delta q = delta (R log(nu))

    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero. dq is computed using numpy not jnp.numpy. If you use jnp, you might observe a significant residual because of the float32 truncation error.

    """
    if(nu_lines.dtype!=np.float64 and warning):
        print("Warning!: nu_lines is not np.float64 but ",nu.dtype)
    if(nu_grid.dtype!=np.float64 and warning):
        print("Warning!: nu_grid is not np.float64 but ",nu.dtype)

    
    R=(len(nu_grid)-1)/np.log(nu_grid[-1]/nu_grid[0]) #resolution
    cont,index=npgetix(nu_lines,nu_grid)
    dq=R*(np.log(nu_grid[1])-np.log(nu_grid[0]))
    
    return jnp.array(cont), jnp.array(index), R, dq


