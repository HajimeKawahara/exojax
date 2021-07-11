"""Line profile computation using Discrete Integral Transform

   * Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekeroma and E.Pannier.
   * This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
   * The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekeroma.

"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
import tqdm
from exojax.spec.ditkernel import folded_voigt_kernel
#from functools import partial

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

@jit
def xsvector3D(nu_lines,sigmaD,gammaL,S,nu_grid,sigmaD_grid,gammaL_grid,dLarray):
    """Cross section vector (DIT/3D version)
    
    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       nu_lines: line center (Nlines)
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

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
       >>> xs=xsvector3D(nu_lines,sigmaD,gammaL,Sij,nus,sigmaD_grid,gammaL_grid,dLarray)

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
    #vk=voigt_kernel(k, sigmaD_grid,gammaL_grid)
    #vk=f1_voigt_kernel(k, sigmaD_grid,gammaL_grid, dnu)
    vk=folded_voigt_kernel(k, sigmaD_grid,gammaL_grid, dLarray)

    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs


@jit
def xsmatrix3D(nu_lines,sigmaDM,gammaLM,SijM,nu_grid,dgm_sigmaD,dgm_gammaL,dLarray):
    """Cross section matrix for xsvector3D (DIT/reduced memory version)

    Args:
       nu_lines: line center (Nlines)
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_sigmaD: DIT Grid Matrix for sigmaD R^(Nlayer, NDITgrid)
       dgm_gammaL: DIT Grid Matrix for gammaL R^(Nlayer, NDITgrid)
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
      
    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
    NDITgrid=jnp.shape(dgm_sigmaD)[1]
    Nline=len(nu_lines)
    Mat=jnp.hstack([sigmaDM,gammaLM,SijM,dgm_sigmaD,dgm_gammaL])
    def fxs(x,arr):
        carry=0.0
        sigmaD=arr[0:Nline]
        gammaL=arr[Nline:2*Nline]
        Sij=arr[2*Nline:3*Nline]
        sigmaD_grid=arr[3*Nline:3*Nline+NDITgrid]
        gammaL_grid=arr[3*Nline+NDITgrid:3*Nline+2*NDITgrid]
        arr=xsvector3D(nu_lines,sigmaD,gammaL,Sij,nu_grid,sigmaD_grid,gammaL_grid,dLarray)
        return carry, arr
    
    val,xsm=scan(fxs,0.0,Mat)
    return xsm
    


@jit
def inc2Dplus(w,fx,y,z,yv,zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 2D+precomputed (memory reduced sum).
    
    Args:
        w: weight (N)
        fx: precomputed nu nc1D
        y: y values (N)
        z: z values (N)
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
    Ngx=len(fx)
    Ngy=len(yv)
    Ngz=len(zv)
    indarry=jnp.arange(Ngy)
    indarrz=jnp.arange(Ngz)
    
    vcl=vmap(Xncf,(0,None,None),0)
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

@jit
def xsvector(nu_ncf,sigmaD,gammaL,S,nu_grid,sigmaD_grid,gammaL_grid, dLarray):
    """Cross section vector (DIT/2D+ version; default)
    
    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       nu_ncf: neibouring contribution function for wavenumber
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    Returns:
       Cross section in the linear nu grid

    Note:
       This function uses the precomputed neibouring contribution function for wavenumber (nu_ncf). Use npnc1D to compute nu_ncf in float64 precision.

    Example:
       >>> nu_ncf=npnc1D(mdbCO.nu_lines,nus)
       >>> Nfold=3
       >>> dLarray=jnp.linspace(1,Nfold,Nfold)/dnus 
       >>> xs=xsvector(nu_ncf,sigmaD,gammaL,Sij,nus,sigmaD_grid,gammaL_grid,dLarray)

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
    val=inc2Dplus(S,nu_ncf,log_sigmaD,log_gammaL,log_sigmaD_grid,log_gammaL_grid)
#    val=inc3D(S,nu_lines,log_sigmaD,log_gammaL,nu_grid,log_sigmaD_grid,log_gammaL_grid)

    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=folded_voigt_kernel(k, sigmaD_grid,gammaL_grid, dLarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

@jit
def xsmatrix(nu_ncf,sigmaDM,gammaLM,SijM,nu_grid,dgm_sigmaD,dgm_gammaL,dLarray):
    """Cross section matrix (DIT/2D+ version)

    Args:
       nu_ncf: neibouring contribution function for wavenumber
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_sigmaD: DIT Grid Matrix for sigmaD R^(Nlayer, NDITgrid)
       dgm_gammaL: DIT Grid Matrix for gammaL R^(Nlayer, NDITgrid)
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
    NDITgrid=jnp.shape(dgm_sigmaD)[1]
    Nline=jnp.shape(SijM)[1]
    Mat=jnp.hstack([sigmaDM,gammaLM,SijM,dgm_sigmaD,dgm_gammaL])
    def fxs(x,arr):
        carry=0.0
        sigmaD=arr[0:Nline]
        gammaL=arr[Nline:2*Nline]
        Sij=arr[2*Nline:3*Nline]
        sigmaD_grid=arr[3*Nline:3*Nline+NDITgrid]
        gammaL_grid=arr[3*Nline+NDITgrid:3*Nline+2*NDITgrid]
        arr=xsvector(nu_ncf,sigmaD,gammaL,Sij,nu_grid,sigmaD_grid,gammaL_grid,dLarray)
        return carry, arr
    
    val,xsm=scan(fxs,0.0,Mat)
    return xsm

def set_ditgrid(x,res=0.1,adopt=True):
    """
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

def make_dLarray(Nfold,dnu):
    """compute dLarray for the DIT folding
    
    Args:
       Nfold: # of the folding
       dnu: linear wavenumber grid interval

    Returns:
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    """
    dLarray=jnp.linspace(1,Nfold,Nfold)/dnu                
    return dLarray

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

def autoNfold(sigma,dnu,pdit=1.5):
    """ determine an adequate Nfold

    Args:
       sigma: sigma for the voigt or gaussian    
       dnu: linear wavenumber grid interval
       pdit: threshold for DIT folding to x=pdit*sigma

    Returns:
       relres: relative resolution of wavenumber grid
       Nfold: suggested Nfold

    Note:
       In DIT w/ folding, we fold the profile to x = 1/dnu * (Nfold + 1/2). We want x > p*sigma, where sigma is sigma in Gaussian or its equivalence of the VOigt profile (0.5*gammaL+sqrt(0.25*gammaL**2+sigmaD**2)). Then, we obtain Nfold > p*dnu/sigma - 1/2 > 0. The relative resolution to the line width is defined by relres = sigma/dnu.

    """
    relres=sigma/dnu
    Nfold=np.max([int(pdit/relres-0.5),1])
    return relres, Nfold

if __name__ == "__main__":

    from exojax.spec import xsection
    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.spec import moldb
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

#    xv=np.linspace(0,1,11) #grid
#    x=np.array([0.23,0.34])
#    print(npnc1D(x,xv))
#    import sys
#    sys.exit()
    
    nus=np.linspace(1900.0,2300.0,80000,dtype=np.float64) 
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
    sigmaD_grid=set_ditgrid(sigmaD,res=0.1)
    gammaL_grid=set_ditgrid(gammaL,res=0.2)


    nu_ncf=npnc1D(mdbCO.nu_lines,nus)
    print(np.shape(nu_ncf))
    xs=xsvector(nu_ncf,mdbCO.nu_lines,sigmaD,gammaL,Sij,nus,sigmaD_grid,gammaL_grid)

    plt.plot(nus,xs)
#    plt.yscale("log")
    plt.show()
