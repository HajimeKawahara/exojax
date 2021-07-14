"""Line profile computation using Discrete Integral Transform

   * Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekeroma and E.Pannier.
   * This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
   * The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekeroma.
   * See also `DIT for non evenly-spaced linear grid <https://github.com/dcmvdbekerom/discrete-integral-transform/blob/master/demo/discrete_integral_transform_log.py>`_ by  D.C.M van den Bekeroma, as a reference of this code.

"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import folded_voigt_kernel_logst

@jit
def Xncf(i,x,xv):
    """neighbouring contribution function for index i.  
    
    Args:
        i: index 
        x: x value
        xv: x-grid
            
    Returns:
        neighbouring contribution function of x to the i-th component of the array with the same dimension as xv.
            
    """
    indarr=jnp.arange(len(xv))
    pos = jnp.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = pos-index
    f=jnp.where(index==i,1.0-cont,0.0)
    g=jnp.where(index+1==i,cont,0.0)
    return f+g


@jit
def inc2D(w,x,y,xv,yv):
    """integrated neighbouring contribution function for 2D (memory reduced sum).
    
    Args:
        w: weight (N)
        x: x values (N)
        y: y values (N)
        xv: x grid
        yv: y grid
            
    Returns:
        integrated neighbouring contribution function
        
    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n, 
        where w_n is the weight, fx_n and fy_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum
        
    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> val=inc2D(w,x,y,xv,yv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc2D(x,y,xv,yv)*w,axis=2)        
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.196106e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 1.6135311e-05 %
    """
    Ngx=len(xv)
    Ngy=len(yv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    vcl=vmap(Xncf,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    fxy_w=jnp.vstack([fx,fy,w]).T
    
    def fsum(x,arr):
        null=0.0
        fx=arr[0:Ngx]
        fy=arr[Ngx:Ngx+Ngy]
        w=arr[Ngx+Ngy]
        val=x+w*fx[:,None]*fy[None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy))
    val,null=scan(fsum,init0,fxy_w)
    return val


def npnc1D(x,xv):
    """numpy version of neighbouring contribution for 1D.
    
    Args:
        x: x value
        xv: x grid
            
    Returns:
        neighbouring contribution function
        

    """
    indarr=np.arange(len(xv))
    pos = np.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = pos-index

    def npXncf(i,x,xv):
        f=np.where(index==i,1.0-cont,0.0)
        g=np.where(index+1==i,cont,0.0)
        return f+g
    vcl=[]
    for i in tqdm.tqdm(indarr):
        vcl.append(npXncf(i,x,xv))
    vcl=jnp.array(np.array(vcl))
                 
    return vcl
    
#            xsv=modit.xsvector(dfnu_lines,nsigmaD,gammaL,Sij,dfnus,gammaL_grid,dLarray,dv_lines,dv)
@jit
def xsvector(nu_lines,nsigmaD,gammaL,S,nu_grid,ngammaL_grid,dLarray,dv_lines,dv_grid):
    """Cross section vector (DIT/3D version)
    
    The original code is rundit_fold_logredst in [addit package](https://github.com/HajimeKawahara/addit). DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       nu_lines: line center (Nlines)
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       dv_lines: delta wavenumber for lines i.e. nu_lines/R
       dv_grid: delta wavenumber for nu_grid i.e. nu_grid/R

    Returns:
       Cross section in the linear nu grid

    Note:
       This version uses inc3D. So, nu grid is also auto-differentiable. However, be careful for the precision of wavenumber grid, because inc3D uses float32 in JAX/GPU. For instance, consider to use dfnus=nus-np.median(nus) and dfnu_lines=mdbCO.nu_lines-np.median(nus) instead of nus (nu_grid) and nu_lines, to mitigate the problem. 

    Example:
       >>> dfnus=nus-np.median(nus)
       >>> dfnu_lines=nu_lines-np.median(nus)
       >>> dnus=nus[1]-nus[0]
       >>> Nfold=3
       >>> dLarray=jnp.linspace(1,Nfold,Nfold)/dnus 

    """

    Ng_nu=len(nu_grid)
    Ng_gammaL=len(ngammaL_grid)

    ngammaL=gammaL/dv_lines
    log_nstbeta=jnp.log(nsigmaD)
    log_ngammaL=jnp.log(ngammaL)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu,1)
    val=inc2D(S,nu_lines,log_ngammaL,nu_grid,log_ngammaL_grid)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=folded_voigt_kernel_logst(k, log_nstbeta,log_ngammaL_grid,dLarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dv_grid
    
    return xs
