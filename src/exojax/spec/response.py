import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def rigidrot(nus,F0,vsini,u1=0.0,u2=0.0):
    """Apply the Rotation response tp a spectrum F using jax.lax.scan

    Args:
        nus: wavenumber
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

    if False:
        N=1000
        varr=jnp.linspace(-100,100,N)
    
        fig=plt.figure()
        for j,beta in enumerate([0.1,1.0,3.0,10.0,30.0]):
            ax=fig.add_subplot(5,1,j+1)
            Mrtm=kernel_rtm(varr, 30.0, beta, 0.0, 0.0,0.0)
            M2D=kernelRTM(varr, 0.0, 30.0, 0.0, 0.0, beta)
            plt.plot(varr,M2D,ls="dashed",color="C1")
            plt.plot(varr,Mrtm,ls="dashed",color="C2")
            
        plt.legend()
        plt.show()

    #grid for F0
    N=1000
    wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
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
    zeta=0.0
    
    ##old
    c=299792.458
#    dvmatx=jnp.array(c*np.log(nusd[:,None]/nus[None,:]))
#    Fr=responseRTM(dvmatx,F0,vsini_in,beta,RV,u1,u2,zeta)

    #new
    Frot=rigidrot(nus,F0,vsini_in,u1,u2)
    Fg=ipgauss(nus,nusd,Frot,beta,RV)
#    plt.plot(wav,F0)
#    plt.plot(wavd,Fr,".")
    plt.plot(wav,Frot)
    plt.plot(wavd,Fg)

#    plt.show()
