"""Line profile computation using Discrete Integral Transform

   * Line profile computation of [Discrete Integral Transform](https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049) for rapid spectral synthesis, originally proposed by D.C.M van den Bekeroma and E.Pannier.
   * This module consists of selected functions in [addit package](https://github.com/HajimeKawahara/addit).

"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan


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
def inc3D(w,x,y,z,xv,yv,zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 3D (memory reduced sum).
    
    Args:
        w: weight (N)
        x: x values (N)
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
        
    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> zv=jnp.linspace(0,1,17) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> z=np.random.rand(N)
        >>> val=inc3D(w,x,y,z,xv,yv,zv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc3D(x,y,z,xv,yv,zv)*w,axis=3)
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.520862e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 8.418057e-06 %
    """
    Ngx=len(xv)
    Ngy=len(yv)
    Ngz=len(zv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    indarrz=jnp.arange(Ngz)
    
    vcl=vmap(Xncf,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    fz=vcl(indarrz,z,zv) # Ngz x N memory

    fxyz_w=jnp.vstack([fx,fy,fz,w]).T
    def fsum(x,arr):
        null=0.0
        fx=arr[0:Ngx]
        fy=arr[Ngx:Ngx+Ngy]
        fz=arr[Ngx+Ngy:Ngx+Ngy+Ngz]
        w=arr[Ngx+Ngy+Ngz]
        val=x+w*fx[:,None,None]*fy[None,:,None]*fz[None,None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy,Ngz))
    val,null=scan(fsum,init0,fxyz_w)
    return val

def voigt_kernel(k, beta,gammaL):
    """Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugated of wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """
    val=(jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]
    return jnp.exp(-2.0*val)

@jit
def xsvector(nu_lines,sigmaD,gammaL,S,nu_grid,sigmaD_grid,gammaL_grid):
    """Cross section vector (DIT version)
    
    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       nu_lines: line center (Nlines)
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the linear nu grid

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
    val=inc3D(S,nu_lines,log_sigmaD,log_gammaL,nu_grid,log_sigmaD_grid,log_gammaL_grid)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=voigt_kernel(k, sigmaD_grid,gammaL_grid)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    #F0=jnp.fft.irfft(fftvalsum)[:Ng_nu]
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

@jit
def xsmatrix(nu_lines,sigmaDM,gammaLM,SijM,nu_grid,sigmaD_grid,gammaL_grid):
    """Cross section matrix (DIT version)

    Args:
       nu_lines: line center (Nlines)
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid

    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
#    xsvector(S,nu_lines,sigmaD,gammaL,nu_grid,sigmaD_grid,gammaL_grid):
    return vmap(xsvector,(None,0,0,0,None,None,None))(nu_lines,sigmaDM,gammaLM,SijM,nu_grid,sigmaD_grid,gammaL_grid)

def set_ditgrid(x,res=1,adopt=True):
    """
    Args:
        x: simgaD or gammaL array (Nline)
        res: grid resolution. res=1 (defaut) means a grid point per digit
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

if __name__ == "__main__":

    from exojax.spec import xsection
    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.spec import moldb
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    nus=np.linspace(1900.0,2300.0,200000,dtype=np.float64) 
    mdbCO=moldb.MdbHit('05_hit12.par',nus)
    Mmol=28.010446441149536 # molecular weight
    Tfix=1000.0 # we assume T=1000K
    Pfix=1.e-3 # we compute P=1.e-3 bar
    Ppart=Pfix #partial pressure of CO. here we assume a 100% CO atmosphere.
    qt=mdbCO.Qr_layer_HAPI([Tfix])[0]
    Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, \
                          mdbCO.gamma_air, mdbCO.gamma_self) \
                          + gamma_natural(mdbCO.A)
    # thermal doppler sigma
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    sigmaD_grid=set_ditgrid(sigmaD,res=1)
    gammaL_grid=set_ditgrid(gammaL,res=1)

    xs=xsvector(mdbCO.nu_lines,sigmaD,gammaL,Sij,nus,sigmaD_grid,gammaL_grid)
    plt.plot(nus,xs)
#    plt.yscale("log")
    plt.show()
