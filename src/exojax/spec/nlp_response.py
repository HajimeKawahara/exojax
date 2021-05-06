"""NLP (Nu-Logspace-Parameterization) response

   * nlp_response is a response operation for the wavenumber grid spaced evenly on a log scale.
   * nlp_response is faster than response.py when N >~ 10000, where N is the dimension of the wavenumber grid.
   * nlp_response uses jax.numpy.convolve, therefore, convolve in cuDNN. 


"""

import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def nlp_rigidrot(nus,F0,varr_kernel,vsini,u1=0.0,u2=0.0):
    """(NLP only) Apply the Rotation response to a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)


    """
    x=varr_kernel/vsini
    x2=x*x
    kernel=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2*2.0*u2*x2),0.0)
    kernel=kernel/jnp.sum(kernel,axis=0)
    F=jnp.convolve(F0,kernel,mode="same")

    return F

@jit
def ipgauss(nus,nusd,F0,beta,RV):
    """Apply the Gaussian IP response tp a spectrum F using jax.lax.scan

    Args:
        nus: input wavenumber
        nusd: wavenumber in an IP frame
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    

    Return:
        response-applied spectrum (F)


    """

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nusd[None,:]/nus[:,None]))
    kernel=jnp.exp(-(dvmat+RV)**2/(2.0*beta**2))
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0
    return F



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exojax.spec import response
    import time

    #grid for F0
    N=100000
    wav=np.logspace(np.log10(22900),np.log10(23000),N,dtype=np.float64) #NLP
    nus=1.e8/wav[::-1]
    
    #grid for F
    M=900
    wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA        
    nusd=1.e8/wavd[::-1]
    
    #dv matrix
    F0=np.ones(N)
    F0[500]=0.5
    vsini_in=10.0
    beta=5.0
    RV=0.0
    u1=0.0
    u2=0.0

    #new
    c=299792.458
    dv=c*(np.log(nus[1])-np.log(nus[0]))
    Nv=int(vsini_in/dv)+1
    vlim=Nv*dv
    Nkernel=2*Nv+1
    varr_kernel=jnp.linspace(-vlim,vlim,Nkernel)
    Frot=nlp_rigidrot(nus,F0,varr_kernel,vsini_in,u1,u2)
    Frotc=response.rigidrot(nus,F0,vsini_in,u1,u2)

    ts=time.time()
    for i in range(0,100):
        Frot=nlp_rigidrot(nus,F0,varr_kernel,vsini_in,u1,u2)
    Frot.block_until_ready()
    te=time.time()
    print("NLP rotation",te-ts,"sec")

    ts=time.time()
    for i in range(0,100):
        Frotc=response.rigidrot(nus,F0,vsini_in,u1,u2)
    Frotc.block_until_ready()
    te=time.time()
    print("normal rotation",te-ts,"sec")

    ts=time.time()
    for i in range(0,100):
        Frot=nlp_rigidrot(nus,F0,varr_kernel,vsini_in,u1,u2)
    Frot.block_until_ready()
    te=time.time()
    print("NLP rotation",te-ts,"sec")

    ts=time.time()
    for i in range(0,100):
        Frotc=response.rigidrot(nus,F0,vsini_in,u1,u2)
    Frotc.block_until_ready()
    te=time.time()
    print("normal rotation",te-ts,"sec")

    
#    Fg=ipgauss(nus,nusd,Frot,beta,RV)
#    plt.plot(wav,F0)
#    plt.plot(wavd,Fr,".")
    plt.plot(wav,F0)
    plt.plot(wav,Frot)
    plt.plot(wav,Frotc,ls="dashed")
    plt.ylim(0.8,1.03)
    plt.xlim(22940,22960)
    plt.savefig("res.png")
#    plt.show()
