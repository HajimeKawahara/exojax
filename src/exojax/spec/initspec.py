""" Initialization for opacity computation. The functions in this module are a wrapper for initialization processes for opacity computation.


"""
import jax.numpy as jnp
import numpy as np
from exojax.spec.dit import npgetix
from exojax.spec.make_numatrix import make_numatrix0
from exojax.utils.instfunc import resolution_eslog

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

def init_dit(nu_lines,nu_grid, warning=False):
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
    warn_dtype64(nu_lines,warning,tag="nu_lines")
    warn_dtype64(nu_grid,warning,tag="nu_grid")
    
    cont,index=npgetix(nu_lines,nu_grid)
    dnu=nu_grid[1]-nu_grid[0]
    pmarray=np.ones(len(nu_grid)+1)
    pmarray[1::2]=pmarray[1::2]*-1

    return jnp.array(cont), jnp.array(index), pmarray

def init_modit(nu_lines,nu_grid, warning=False):
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
    warn_dtype64(nu_lines,warning,tag="nu_lines")
    warn_dtype64(nu_grid,warning,tag="nu_grid")

    R=resolution_eslog(nu_grid)
    cont,index=npgetix(nu_lines,nu_grid)
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
    warn_dtype64(nu_lines,warning,tag="nu_lines")
    warn_dtype64(nu_grid,warning,tag="nu_grid")

    R=resolution_eslog(nu_grid)
    cont,index=npgetix(nu_lines,nu_grid)
    dq=R*(np.log(nu_grid[1])-np.log(nu_grid[0]))
    
    return jnp.array(cont), jnp.array(index), R, dq


def warn_dtype64(arr,warning,tag=""):
    """check arr's dtype

    Args:
       arr: input array
       warning: True/False
       tag: 

    """
    if(arr.dtype!=np.float64 and warning):
        print("Warning!: "+tag+" is not np.float64 but ",arr.dtype)