""" Initialization for opacity computation. The functions in this module are a wrapper for initialization processes for opacity computation.


"""
import jax.numpy as jnp
import numpy as np
from exojax.spec.dit import npgetix
from exojax.spec.make_numatrix import make_numatrix0
from exojax.spec.dit import make_dLarray

def init_lpf(nu_lines,nu_grid):
    """Initialization for LPF

    Args:
        nu_lines: wavenumber list of lines [Nline]
        nu_grid: wavenumenr grid [Nnugrid]

    Returns:
       numatrix [Nline,Nnu]

    """
    numatrix=make_numatrix0(nu_grid,nu_lines,warning=True)
    return numatrix

def init_dit(nu_lines,nu_grid,Nfold=1):
    """Initialization for DIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline]
        nu_grid: wavenumenr grid [Nnugrid]
        Nfold: number of folding

    Returns:
        cont (contribution) jnp.array
        index (index) jnp.array
        dLarray: folding array

    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero.
    """
    cont,index=npgetix(nu_lines,nu_grid)
    dnu=nu_grid[1]-nu_grid[0]
    dLarray=make_dLarray(Nfold,dnu)
    return jnp.array(cont), jnp.array(index), dLarray

def init_modit(nu_lines,nu_grid,Nfold=1):
    """Initialization for MODIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline]
        nu_grid: wavenumenr grid [Nnugrid]
        Nfold: number of folding

    Returns:
        cont: (contribution) jnp.array
        index: (index) jnp.array
        R: spectral resolution
        dLarray: folding array

    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero.

    """

    R=(len(nu_grid)-1)/np.log(nu_grid[-1]/nu_grid[0]) #resolution
    cont,index=npgetix(nu_lines,nu_grid)
    dLarray=make_dLarray(Nfold,1)
    return jnp.array(cont), jnp.array(index), R, dLarray

def init_redit(nu_lines,nu_grid):
    """Initialization for REDIT. i.e. Generate nu contribution and index for the line shape density (actually, this is a numpy version of getix)

    Args:
        nu_lines: wavenumber list of lines [Nline]
        nu_grid: wavenumenr grid [Nnugrid]

    Returns:
        cont (contribution) jnp.array
        index (index) jnp.array
        R: spectral resolution
        dq: delta q = delta (R log(nu))

    Note:
       cont is the contribution for i=index. 1 - cont is the contribution for i=index+1. For other i, the contribution should be zero. dq is computed using numpy not jnp.numpy. If you use jnp, you might observe a significant residual because of the float32 truncation error.

    """
    R=(len(nu_grid)-1)/np.log(nu_grid[-1]/nu_grid[0]) #resolution
    cont,index=npgetix(nu_lines,nu_grid)
    dq=R*(np.log(nu_grid[1])-np.log(nu_grid[0]))
    
    return jnp.array(cont), jnp.array(index), R, dq
