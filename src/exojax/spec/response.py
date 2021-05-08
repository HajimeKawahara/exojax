"""response

   * input nus/wav should be spaced evenly on a log scale (ESLOG).
   * response is a response operation for the wavenumber grid spaced evenly on a log scale.
   * rigidrot2 and ipgauss2 are faster than default when N >~ 10000, where N is the dimension of the wavenumber grid.
   * response uses jax.numpy.convolve, therefore, convolve in cuDNN. 


"""

import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def rigidrot(nus,F0,vsini,u1=0.0,u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)


    """
    c=299792.458
    dvmat=jnp.array(c*jnp.log(nus[None,:]/nus[:,None]))
    x=dvmat/vsini
    x2=x*x
    kernel=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2*2.0*u2*x2),0.0)
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0

    return F

@jit
def ipgauss(nus,F0,beta):
    """Apply the Gaussian IP response to a spectrum F using jax.lax.scan

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)


    """

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nus[None,:]/nus[:,None]))    
    kernel=jnp.exp(-(dvmat)**2/(2.0*beta**2))
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0
    return F

@jit
def ipgauss_sampling(nusd,nus,F0,beta,RV):
    """Apply the Gaussian IP response + sampling to a spectrum F 

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)

    Return:
        response-applied spectrum (F)


    """

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nusd[None,:]/nus[:,None]))    
    kernel=jnp.exp(-(dvmat+RV)**2/(2.0*beta**2))    
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0
    return F



@jit
def rigidrot2(nus,F0,varr_kernel,vsini,u1=0.0,u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber, evenly log-spaced
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
def ipgauss2(nus,F0,varr_kernel,beta):
    """Apply the Gaussian IP response to a spectrum F

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)


    """
    x=varr_kernel/beta
    kernel=jnp.exp(-x*x/2.0)
    kernel=kernel/jnp.sum(kernel,axis=0)
    F=jnp.convolve(F0,kernel,mode="same")

    return F

@jit
def sampling(nusd,nus,F,RV):
    """Sampling w/ RV

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber
        F: input spectrum
        RV: radial velocity (km/s)

    Returns:
       sampled spectrum
    
    """
    c=299792.458
    return jnp.interp(nusd*(1.0+RV/c),nus,F)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    #grid for F0
    N=1000
    wav=np.logspace(np.log10(22900),np.log10(23000),N,dtype=np.float64) #NLP
    nus=1.e8/wav[::-1]

    wavn=wav
    nusn=nus
    # one can see deviation if using non-ESLOG form 
    #NN=1000
    #wavn=np.linspace(22900,23000,NN,dtype=np.float64) 
    #nusn=1.e8/wavn[::-1]
    
    #dv matrix
    F0=np.ones(N)
    F0[504]=0.5
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
    Frot=rigidrot2(nus,F0,varr_kernel,vsini_in,u1,u2)                  
       
    maxp=5.0 #5sigma
    Nv=int(maxp*beta/dv)+1
    vlim=Nv*dv
    Nkernel=2*Nv+1
    varr_kernel=jnp.linspace(-vlim,vlim,Nkernel)
    Fgrot=ipgauss2(nus,Frot,varr_kernel,beta)                      

    Frotc=rigidrot(nusn,F0,vsini_in,u1,u2)

    #grid for F
    M=450
    wavd=np.linspace((22900),(23000),M,dtype=np.float64) #NLP
    nusd=1.e8/wavd[::-1]
    
    #    Fgrotd=jnp.interp(nusd,nus,Fgrot)
    Fgrotd=sampling(nusd,nus,Fgrot,RV)
    Fgrotc=ipgauss_sampling(nusd,nusn,Frotc,beta,RV)

    #    nusd=np.logspace(np.log10(1.e8/23000),np.log10(1.e8/22900),M,dtype=np.float64) #NLP
    #    wavd=1.e8/nusd[::-1]
    
    
    if True:
        #    plt.plot(wav,F0)
        #    plt.plot(wavd,Fr,".")
        plt.plot(wav[::-1],F0)
        plt.plot(wav[::-1],Frot,".",color="C0",lw=1)
        plt.plot(wav[::-1],Fgrot,".",color="C2",lw=1)
        plt.plot(wavd[::-1],Fgrotd,"+",color="C4",lw=2)
        
        plt.plot(wav[::-1],Frot,alpha=0.3,color="C0",lw=1)
        plt.plot(wav[::-1],Fgrot,alpha=0.3,color="C2",lw=1)
        
        plt.plot(wavn[::-1],Frotc,".",color="C1",lw=1)
        plt.plot(wavn[::-1],Frotc,alpha=0.3,color="C1",lw=1)

        plt.plot(wavd[::-1],Fgrotc,"s",color="C3",lw=3)
        plt.plot(wavd[::-1],Fgrotc,alpha=0.3,color="C3",lw=1)
                
        plt.ylim(0.95,1.03)
        plt.xlim(22947,22953)
        #plt.savefig("res.png")
        plt.show()
        import sys
        sys.exit()
        
    else:
        #TIME CHALLAENGE
        ts=time.time()
        for i in range(0,100):
            Frot=rigidrot2(nus,F0,varr_kernel,vsini_in,u1,u2)
            Fgrot=ipgauss2(nus,Frot,varr_kernel,beta)                      
            Fgrotd=sampling(nusd,nus,Fgrot,RV)
            
        Fgrot.block_until_ready()
        te=time.time()
        print("rotation 2",te-ts,"sec")

        ts=time.time()
        for i in range(0,100):
            Frotc=rigidrot(nus,F0,vsini_in,u1,u2)
            Fgrotc=ipgauss(nus,Frotc,beta)
            Fgrotcd=sampling(nusd,nus,Fgrotc,RV)

        Fgrotc.block_until_ready()
        te=time.time()
        print("rotation",te-ts,"sec")

        ts=time.time()
        for i in range(0,100):
            Frot=rigidrot2(nus,F0,varr_kernel,vsini_in,u1,u2)
            Fgrot=ipgauss2(nus,Frot,varr_kernel,beta)                      
            Fgrotd=sampling(nusd,nus,Fgrot,RV)
            
        Fgrot.block_until_ready()
        te=time.time()
        print("rotation 2",te-ts,"sec")
        
        ts=time.time()
        for i in range(0,100):
            Frotc=rigidrot(nus,F0,vsini_in,u1,u2)
            Fgrotc=ipgauss(nus,Frotc,beta)
            Fgrotcd=sampling(nusd,nus,Fgrotc,RV)
        
        Fgrotc.block_until_ready()
        te=time.time()
        print("rotation",te-ts,"sec")

