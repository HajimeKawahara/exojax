import jax.numpy as jnp
from jax import jit
from jax.lax import scan
import numpy as np
from exojax.special.j0 import j0



xi = jnp.array([-0.9914553711208126392069,-0.9491079123427585245262,-0.8648644233597690727897,-0.7415311855993944398639,-0.5860872354676911302941,-0.4058451513773971669066,-0.2077849550078984676007,0,0.2077849550078984676007,0.4058451513773971669066,0.5860872354676911302941,0.7415311855993944398639,0.8648644233597690727897,0.9491079123427585245262,0.9914553711208126392069])
wi = jnp.array([0.0229353220105292249637,0.0630920926299785532907,0.1047900103222501838399,0.140653259715525918745,0.1690047266392679028266,0.1903505780647854099133,0.2044329400752988924142,0.209482141084727828013,0.2044329400752988924142,0.190350578064785409913,0.1690047266392679028266,0.1406532597155259187452,0.10479001032225018384,0.0630920926299785532907,0.02293532201052922496373])

@jit
def kernelRTM(varr, zeta, vsini, u1, u2, beta):
    """Response Kernel including rotation, a Gaussian broadening (IP+microturbulence), radial-tangential turbulence,   

    Args:
       varr: velocity array
       zeta: macroturbulence disturbunce (Gray 2005)
       vsini: V sini for rotation
       u1: Limb-darkening coefficient 1
       u2: Limb-darkening coefficient 2
       beta: STD of a Gaussian broadening (IP+microturbulence)

    Returns:
       normalized response kernel

    Note:
       This function uses the Gauss-Kronrod method for the integral, originally developed by K. Masuda. See Masuda and Hirano (2021) ApJL  910, L17, for the details. The unit of varr, zeta, vsini, and beta should be same, such as km/s. A small beta (<~1 km/s) induces an instability of the kernel. 

    """
    dv = varr[:,None] - vsini * xi
    dv2 = dv * dv
    ominx2 = 1. - xi * xi
    a, b = -jnp.sqrt(ominx2), jnp.sqrt(ominx2)
    d = 0.5 * (b - a)
    xs = d * xi[:,None]
    cosg2 = ominx2 - xs * xs
    cosg = jnp.sqrt(cosg2)
    sing2 = 1. - cosg2
    sigma2_cos = beta*beta + 0.5*zeta*zeta*cosg2
    sigma2_sin = beta*beta + 0.5*zeta*zeta*sing2
    ys = 0.5 * (jnp.exp(-0.5*dv2[:,None]/sigma2_cos)/jnp.sqrt(2*jnp.pi*sigma2_cos) + jnp.exp(-0.5*dv2[:,None]/sigma2_sin)/jnp.sqrt(2*jnp.pi*sigma2_sin))
    ys *= 1. - (1. - cosg) * (u1 + u2 * (1. - cosg))
    ys /= jnp.pi * (1. - u1/3. - u2/6.)
    xintegrand = d * jnp.sum(wi[:,None] * ys, axis=1)
    k=jnp.sum(wi * xintegrand, axis=1)
    ksum=jnp.sum(k)
    return k/ksum


def responseRTM(dvmat,F0,vsini,beta,RV,u1=0.0,u2=0.0,zeta=0.0):
    """Apply the RTM response tp a spectrum F using jax.lax.scan

    Args:
        dvmat: velocity matrix (jnp.array)
        F0: original spectrum (F0)
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity    
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2
        zeta: macroturbulence disturbunce (Gray 2005)

    Return:
        response-applied spectrum (F)


    """
    def respense_fscan(carry,varr):
        """function for a scanning response

        Args:
           carry: dummy
           varr: velocity array

        Returns:
           dummy, kernel multiplied F       

        Note:
           This function computes a single point of F(nu).

        """
        Fr=jnp.sum(F0*kernelRTM(varr, zeta, vsini, u1, u2, beta))
        return carry,Fr

    c=299792.458
    car,F=scan(respense_fscan,0.0,dvmat-RV)

    return F


@jit
def kernel_rigidrot(varr,vsini,u1,u2):
    """Response Kernel of a rigit rotation

    Args:
       varr: velocity array
       vsini: V sini for rotation
       u1: Limb-darkening coefficient 1
       u2: Limb-darkening coefficient 2

    Returns:
       normalized response kernel

    """
    x=varr/vsini
    x2=x*x
    m=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2*2.0*u2*x2),0.0)
    return m/jnp.sum(m)


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
    def rigidrot_fscan(carry,varr):
        """scanning response

        Args:
           carry: dummy
           varr: velocity array

        Returns:
           dummy, kernel multiplied F       

        Note:
           This function computes a single point of F(nu) iteratively. The scanning is done across the wavenumber direction of F(nu).


        """
        Fr=jnp.sum(F0*kernel_rigidrot(varr,vsini,u1,u2))
        return carry,Fr

    
    c=299792.458
    dvmat0=jnp.array(c*jnp.log(nus[:,None]/nus[None,:]))

    car,F=scan(rigidrot_fscan,0.0,dvmat0)

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
    @jit
    def ipgauss_fscan(carry,varr):
        """scanning response

        Args:
           carry: dummy
           varr: velocity array

        Returns:
           dummy, kernel multiplied F       

        Note:
           This function computes a single point of F(nu) iteratively. The scanning is done across the wavenumber direction of F(nu).


        """
        kernel=jnp.exp(-varr**2/(2.0*beta**2))
        kernel=kernel/jnp.sum(kernel)
        Fr=jnp.sum(F0*kernel)
        return carry,Fr

    c=299792.458
    dvmat=jnp.array(c*jnp.log(nusd[:,None]/nus[None,:]))
    car,F=scan(ipgauss_fscan,0.0,dvmat+RV)

    return F

@jit
def rigidrotm(nus,F0,vsini,u1=0.0,u2=0.0):
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
#    kernel=jnp.exp(-(dvmat+RV)**2/(2.0*beta**2))
    x=(dvmat+RV)/vsini
    x2=x*x
    kernel=jnp.where(x2<1.0,jnp.pi/2.0*u1*(1.0 - x2) - 2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2*2.0*u2*x2),0.0)
    kernel=kernel/jnp.sum(kernel,axis=0) #axis=N
    F=kernel.T@F0

    return F

@jit
def ipgaussm(nus,nusd,F0,beta,RV):
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
    dvmatx=jnp.array(c*np.log(nusd[:,None]/nus[None,:]))
    Fr=responseRTM(dvmatx,F0,vsini_in,beta,RV,u1,u2,zeta)

    #new
    Frot=rigidrotm(nus,F0,vsini_in,u1,u2)
    Fg=ipgaussm(nus,nusd,Frot,beta,RV)

    plt.plot(wav,F0)
    plt.plot(wavd,Fr,".")

    plt.plot(wav,Frot)
    plt.plot(wavd,Fg)

    plt.show()
