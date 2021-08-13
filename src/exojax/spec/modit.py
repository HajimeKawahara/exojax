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
from jax.ops import index_add
from jax.ops import index as joi
from exojax.spec.dit import getix

@jit
def inc2D_givenx(a,w,cx,ix,y,yv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 2D (memory reduced sum) but using given contribution and index for x .
    
    Args:
        a: lineshape density array (jnp.array)
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
        In this function, we use jax.lax.scan to compute the sum

    """

    cy,iy=getix(y,yv)

    a=index_add(a,joi[ix,iy],w*(1-cx)*(1-cy))
    a=index_add(a,joi[ix,iy+1],w*(1-cx)*cy)
    a=index_add(a,joi[ix+1,iy],w*cx*(1-cy))
    a=index_add(a,joi[ix+1,iy+1],w*cx*cy)

    return a




@jit
def xsvector(cnu,indexnu,R,dLarray,nsigmaD,ngammaL,S,nu_grid,ngammaL_grid):
    """Cross section vector (DIT/3D version)
    
    The original code is rundit_fold_logredst in [addit package](https://github.com/HajimeKawahara/addit). DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the linear nu grid


    """

    Ng_nu=len(nu_grid)
    Ng_gammaL=len(ngammaL_grid)

    log_nstbeta=jnp.log(nsigmaD)
    log_ngammaL=jnp.log(ngammaL)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu,1)    
    lsda=jnp.zeros((len(nu_grid),len(ngammaL_grid))) #LSD array
    Slsd=inc2D_givenx(lsda, S,cnu,indexnu,log_ngammaL,log_ngammaL_grid) #Lineshape Density
    Sbuf=jnp.vstack([Slsd,jnp.zeros_like(Slsd)])
    til_Slsd = jnp.fft.rfft(Sbuf,axis=0)
    
    til_Voigt=folded_voigt_kernel_logst(k, log_nstbeta,log_ngammaL_grid,dLarray)
    fftvalsum = jnp.sum(til_Slsd*til_Voigt,axis=(1,))
    
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nu_grid
    
    return xs


@jit
def xsmatrix(cnu,indexnu,R,dLarray,nsigmaDl,ngammaLM,SijM,nu_grid,dgm_ngammaL):
    """Cross section matrix for xsvector (MODIT)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       dLarray: ifold/dnu (ifold=1,..,Nfold) array
       nu_lines: line center (Nlines)
       nsigmaDl: normalized doppler sigma in layers in R^(Nlayer x 1)
       ngammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_ngammaL: DIT Grid Matrix for normalized gammaL R^(Nlayer, NDITgrid)
      
    Return:
       cross section matrix in R^(Nlayer x Nwav)

    """
    NDITgrid=jnp.shape(dgm_ngammaL)[1]
    Nline=len(cnu)
    Mat=jnp.hstack([nsigmaDl,ngammaLM,SijM,dgm_ngammaL])
    def fxs(x,arr):
        carry=0.0
        nsigmaD=arr[0:1]
        ngammaL=arr[1:Nline+1]
        Sij=arr[Nline+1:2*Nline+1]
        ngammaL_grid=arr[2*Nline+1:2*Nline+NDITgrid+1]
        arr=xsvector(cnu,indexnu,R,dLarray,nsigmaD,ngammaL,Sij,nu_grid,ngammaL_grid)
        return carry, arr
    
    val,xsm=scan(fxs,0.0,Mat)
    return xsm

    

def minmax_dgmatrix(x,res=0.1,adopt=True):
    """compute MIN and MAX DIT GRID MATRIX

    Args:                                                                       
        x: gammaL matrix (Nlayer x Nline)                             
        res: grid resolution. res=0.1 (defaut) means a grid point per digit     
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be res exactly. 
                                                                                
    Returns:                                                                    
        minimum and maximum for DIT (dgm_minmax)
                                                                                
    """
    mmax=np.max(np.log10(x),axis=1)
    mmin=np.min(np.log10(x),axis=1)
    Nlayer=np.shape(mmax)[0]
    gm_minmax=[]
    dlog=np.max(mmax-mmin)
    Ng=(dlog/res).astype(int)+2
    for i in range(0,Nlayer):
        lxmin=mmin[i]
        lxmax=mmax[i]
        grid=[lxmin,lxmax]
        gm_minmax.append(grid)
    gm_minmax=np.array(gm_minmax)
    return gm_minmax

def precompute_dgmatrix(set_gm_minmax,res=0.1,adopt=True):
    """Precomputing MODIT GRID MATRIX for normalized GammaL

    Args:
        set_gm_minmax: set of gm_minmax for different parameters [Nsample, Nlayers, 2], 2=min,max
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)  

    """
                      
    lminarray=np.min(set_gm_minmax[:,:,0],axis=0) #min
    lmaxarray=np.max(set_gm_minmax[:,:,1],axis=0)  #max
    dlog=np.max(lmaxarray-lminarray)
    gm=[]
    Ng=(dlog/res).astype(int)+2
    Nlayer=len(lminarray)
    for i in range(0,Nlayer):
        lxmin=lminarray[i]
        lxmax=lmaxarray[i]
        if adopt==False:
            grid=np.logspace(lxmin,lxmin+(Ng-1)*res,Ng)
        else:
            grid=np.logspace(lxmin,lxmax,Ng)
        gm.append(grid)
    gm=np.array(gm)
    return gm

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.spec import rtcheck, moldb
    from exojax.spec.dit import make_dLarray
    from exojax.spec.dit import set_ditgrid
    from exojax.spec.hitran import normalized_doppler_sigma

    nus_modit=np.logspace(np.log10(3000),np.log10(6000.0),1000000,dtype=np.float64)

    mdbCO=moldb.MdbHit('/home/kawahara/exojax/data/CO/05_hit12.par',nus_modit)
    #USE TIPS partition function
    Q296=np.array([107.25937215917970,224.38496958496091,112.61710362499998,\
                   660.22969049609367,236.14433662109374,1382.8672147421873])
    Q1000=np.array([382.19096582031250,802.30952197265628,402.80326733398437,\
                    2357.1041210937501,847.84866308593757,4928.7215078125000])
    qr=Q1000/Q296
    qt=np.ones_like(mdbCO.isoid,dtype=np.float64)
    for idx,iso in enumerate(mdbCO.uniqiso):
        mask=mdbCO.isoid==iso
        qt[mask]=qr[idx]
    
    Mmol=28.010446441149536
    Tref=296.0
    Tfix=1000.0
    Pfix=1.e-3 #
    
    Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    gammaL = gamma_hitran(Pfix,Tfix,Pfix, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self)
    #+ gamma_natural(A) #uncomment if you inclide a natural width
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    
    
    R=(len(nus_modit)-1)/np.log(nus_modit[-1]/nus_modit[0]) #resolution
    dv_lines=mdbCO.nu_lines/R
    nsigmaD=normalized_doppler_sigma(Tfix,Mmol,R)
    ngammaL=gammaL/dv_lines
    ngammaL_grid=set_ditgrid(ngammaL)
    
    Nfold=1
    dLarray=make_dLarray(Nfold,1)
    
    dfnus=nus_modit-np.median(nus_modit) #remove median
    dfnu_lines=mdbCO.nu_lines-np.median(nus_modit) #remove median
    dv=nus_modit/R #delta wavenumber grid
    xs_modit_lp=xsvector(dfnu_lines,nsigmaD,ngammaL,Sij,dfnus,ngammaL_grid,dLarray,dv_lines,dv)
    wls_modit = 100000000/nus_modit

    plt.plot(wls_modit,xs_modit_lp,ls="dashed",color="C1",alpha=0.7,label="MODIT")

    plt.ylim(1.e-27,3.e-19)
    plt.yscale("log")
    tip=20.0
    llow=2300.4
    lhigh=2300.7
    plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
    plt.show()
