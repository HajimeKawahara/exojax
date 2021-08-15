""" Real space evaluation of DIT

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
from exojax.spec.modit import inc2D_givenx
from exojax.spec.lpf import voigt

@jit
def xsvector(cnu,indexnu,R,nsigmaD,ngammaL,S,nu_grid,ngammaL_grid):
    """Cross section vector (REDIT/3D version)
    
    The original code is rundit_fold_logredst in [addit package](https://github.com/HajimeKawahara/addit). DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
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
    lsda=jnp.zeros((len(nu_grid),len(ngammaL_grid))) #LSD array init
    Slsd=inc2D_givenx(lsda, S,cnu,indexnu,log_ngammaL,log_ngammaL_grid) #LSD

    qvector=
    
    Mat=jnp.hstack([log_ngammaL_grid,Slsd])
    def seqconv(x,arr):        
        carry=0.0
        log_ngammaL_each=arr[0]
        se=arr[1:]        
        kernel=voigt(qvector,log_nstbeta,log_ngammaL_each)
        arr=jnp.convolve(se,kernel,mode="same")  
        return carry, arr
    
    val,xsmm=scan(seqconv,0.0,Mat)
    xsm=jnp.sum(xsmm,axis=0)
    
    return xsm

if __name__=="__main__":
    print("test")
    # a_i

    #convolution
    v=jnp.array([1,9,1])
    s=jnp.array([0,0,0,0,1,2,0,0,0,2,0,0,0,0,0,1])
    c=jnp.convolve(s,v,mode="same")
    print(jnp.shape(c),jnp.shape(s))
    print(c)


    #convolution
    va=jnp.array([[1,9,1],[1,3,1]])
    Nkernel=jnp.shape(va)[1]
    sa=jnp.array([[0,0,0,0,1,2,0,0,0,2,0,0,0,0,0,1],\
                  [0,0,1,2,0,0,0,0,0,2,0,0,0,0,1,0]])
    Mat=jnp.hstack([va,sa])
    def seqconv(x,arr):        
        carry=0.0
        ve=arr[0:Nkernel]
        se=arr[Nkernel:]
        arr=jnp.convolve(se,ve,mode="same")  
        return carry, arr
    
    val,xsmm=scan(seqconv,0.0,Mat)
    print(jnp.sum(xsmm,axis=0))
    
#    c=
#    print(jnp.shape(c),jnp.shape(s))
#    print(c)


        
