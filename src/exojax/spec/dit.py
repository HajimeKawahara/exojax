"""Line profile computation using Discrete Integral Transform

   * Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E.Pannier.
   * This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
   * The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekeroma.

"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import fold_voigt_kernel
from jax.ops import index_add
from jax.ops import index as joi

def getix(x,xv):
    """ jnp version of getix

    Args:
        x: x array
        xv: x grid 

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.

    Example:
       
       >>> from exojax.spec.dit import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))    


    """
    indarr=jnp.arange(len(xv))
    pos = jnp.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont,index

def npgetix(x,xv):
    """numpy version of getix

    Args:
        x: x array
        xv: x grid 

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.



    """
    indarr=np.arange(len(xv))
    pos = np.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont,index

@jit
def inc3D_givenx(a,w,cx,ix,y,z,xv,yv,zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 3D (memory reduced sum) but using given contribution and index for x .
    
    Args:
        a: lineshape density array (jnp.array)
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
        In this function, we use jax.lax.scan to compute the sum

    """

    cy,iy=getix(y,yv)
    cz,iz=getix(z,zv)

    a=index_add(a,joi[ix,iy,iz],w*(1-cx)*(1-cy)*(1-cz))
    a=index_add(a,joi[ix,iy+1,iz],w*(1-cx)*cy*(1-cz))
    a=index_add(a,joi[ix+1,iy,iz],w*cx*(1-cy)*(1-cz))
    a=index_add(a,joi[ix+1,iy+1,iz],w*cx*cy*(1-cz))
    a=index_add(a,joi[ix,iy,iz+1],w*(1-cx)*(1-cy)*cz)
    a=index_add(a,joi[ix,iy+1,iz+1],w*(1-cx)*cy*cz)
    a=index_add(a,joi[ix+1,iy,iz+1],w*cx*(1-cy)*cz)
    a=index_add(a,joi[ix+1,iy+1,iz+1],w*cx*cy*cz)

    return a

@jit
def xsvector(cnu,indexnu,pmarray,sigmaD,gammaL,S,nu_grid,sigmaD_grid,gammaL_grid):
    """Cross section vector (DIT/2D+ version; default)
    
    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the linear nu grid

    Note:
       This function uses the precomputed neibouring contribution function for wavenumber (nu_ncf). Use npnc1D to compute nu_ncf in float64 precision.

    """
    Ng_nu=len(nu_grid)
    Ng_sigmaD=len(sigmaD_grid)
    Ng_gammaL=len(gammaL_grid)
    
    log_sigmaD=jnp.log(sigmaD)
    log_gammaL=jnp.log(gammaL)
    
    log_sigmaD_grid = jnp.log(sigmaD_grid)
    log_gammaL_grid = jnp.log(gammaL_grid)
    dnu = (nu_grid[-1]-nu_grid[0])/(Ng_nu-1)
    k = jnp.fft.rfftfreq(2*Ng_nu,dnu)
    
    lsda=jnp.zeros((len(nu_grid),len(log_sigmaD_grid),len(log_gammaL_grid)))
    val=inc3D_givenx(lsda,S,cnu,indexnu,log_sigmaD,log_gammaL,nu_grid,log_sigmaD_grid,log_gammaL_grid)

    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vmax=Ng_nu*dnu
    vk=fold_voigt_kernel(k, sigmaD_grid,gammaL_grid, vmax, pmarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

@jit
def xsmatrix(cnu,indexnu,pmarray,sigmaDM,gammaLM,SijM,nu_grid,dgm_sigmaD,dgm_gammaL):
    """Cross section matrix (DIT/2D+ version)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_sigmaD: DIT Grid Matrix for sigmaD R^(Nlayer, NDITgrid)
       dgm_gammaL: DIT Grid Matrix for gammaL R^(Nlayer, NDITgrid)

    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
    NDITgrid=jnp.shape(dgm_sigmaD)[1]
    Nline=len(cnu)
    Mat=jnp.hstack([sigmaDM,gammaLM,SijM,dgm_sigmaD,dgm_gammaL])
    def fxs(x,arr):
        carry=0.0
        sigmaD=arr[0:Nline]
        gammaL=arr[Nline:2*Nline]
        Sij=arr[2*Nline:3*Nline]
        sigmaD_grid=arr[3*Nline:3*Nline+NDITgrid]
        gammaL_grid=arr[3*Nline+NDITgrid:3*Nline+2*NDITgrid]
        arr=xsvector(cnu,indexnu,pmarray,sigmaD,gammaL,Sij,nu_grid,sigmaD_grid,gammaL_grid)
        return carry, arr
    
    val,xsm=scan(fxs,0.0,Mat)
    return xsm

def ditgrid(x,res=0.1,adopt=True):
    """DIT GRID

    Args:
        x: simgaD or gammaL array (Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. 
               In this case, the grid width does not need to be res exactly.
        
    Returns:
        grid for DIT
        
    """
    lxmin=np.log10(np.min(x))
    lxmax=np.log10(np.max(x))
    dlog=lxmax-lxmin
    Ng=int(dlog/res)+2
    if adopt==False:
        grid=np.logspace(lxmin,lxmin+(Ng-1)*res,Ng)
    else:
        grid=np.logspace(lxmin,lxmax,Ng)
    return grid

def set_ditgrid(x,res=0.1,adopt=True):
    """alias of ditgrid

    Args:
        x: simgaD or gammaL array (Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. 
               In this case, the grid width does not need to be res exactly.
        
    Returns:
        grid for DIT

    """
    return ditgrid(x,res,adopt)
    
def dgmatrix(x,res=0.1,adopt=True):
    """DIT GRID MATRIX 

    Args:
        x: simgaD or gammaL matrix (Nlayer x Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. 
               In this case, the grid width does not need to be res exactly.
        
    Returns:
        grid for DIT (Nlayer x NDITgrid)

    """
    mmax=np.max(np.log10(x),axis=1)
    mmin=np.min(np.log10(x),axis=1)
    Nlayer=np.shape(mmax)[0]
    gm=[]
    dlog=np.max(mmax-mmin)
    Ng=(dlog/res).astype(int)+2
    for i in range(0,Nlayer):
        lxmin=mmin[i]
        lxmax=mmax[i]
        if adopt==False:
            grid=np.logspace(lxmin,lxmin+(Ng-1)*res,Ng)
        else:
            grid=np.logspace(lxmin,lxmax,Ng)
        gm.append(grid)
    gm=np.array(gm)
    return gm

def sigma_voigt(dgm_sigmaD,dgm_gammaL):
    """compute sigma of the Voigt profile
    
    Args:
       dgm_sigmaD: DIT grid matrix for sigmaD
       dgm_gammaL: DIT grid matrix for gammaL

    Returns:
       sigma

    """
    fac=2.*np.sqrt(2.*np.log(2.0))
    fdgm_gammaL=jnp.min(dgm_gammaL,axis=1)*2.0
    fdgm_sigmaD=jnp.min(dgm_sigmaD,axis=1)*fac
    fv=jnp.min(0.5346*fdgm_gammaL+jnp.sqrt(0.2166*fdgm_gammaL**2+fdgm_sigmaD**2))
    sigma=fv/fac
    return sigma

