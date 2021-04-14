import jax.numpy as jnp
from jax import jit


#%%
xi = jnp.array([-0.9914553711208126392069,-0.9491079123427585245262,-0.8648644233597690727897,-0.7415311855993944398639,-0.5860872354676911302941,-0.4058451513773971669066,-0.2077849550078984676007,0,0.2077849550078984676007,0.4058451513773971669066,0.5860872354676911302941,0.7415311855993944398639,0.8648644233597690727897,0.9491079123427585245262,0.9914553711208126392069])
wi = jnp.array([0.0229353220105292249637,0.0630920926299785532907,0.1047900103222501838399,0.140653259715525918745,0.1690047266392679028266,0.1903505780647854099133,0.2044329400752988924142,0.209482141084727828013,0.2044329400752988924142,0.190350578064785409913,0.1690047266392679028266,0.1406532597155259187452,0.10479001032225018384,0.0630920926299785532907,0.02293532201052922496373])

@jit
def rotspectrum(varr, zeta, vsini, u1, u2, beta):
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
    return k


def rotgray(vsini,epsilon):
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    lambda_ratio_sqr = (delta_lambdas / delta_lambda_l) ** 2.0
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator
    kernel = c1 * np.sqrt(1.0 - lambda_ratio_sqr) + c2 * (1.0 - lambda_ratio_sqr)
    return kernel



if __name__ == "__main__":
    from exojax.spec import AutoRT
    import numpy as np 
    import matplotlib.pyplot as plt
    import sys
    from jax.lax import scan
    
    c=300000.0
    N=10000
    M=7500
    wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
    wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA
    
    vsini=15.0
    
    nus=1.e8/wav[::-1]
    nusd=1.e8/wavd[::-1]
    dvmat=c*np.log(nusd[:,None]/nus[None,:])-0.0

#    plt.imshow(mat)
#    plt.show()

    Parr=np.logspace(-8,2,100) #100 pressure layers (10**-8 to 100 bar)
    Tarr = 1500.*(Parr/Parr[-1])**0.02    #some T-P profile
    autort=AutoRT(nus,1.e5,Tarr,Parr)     #g=1.e5 cm/s2
    autort.addmol("ExoMol","CO",0.01)     #mmr=0.01
    F=autort.rtrun()

    #### SCAN
    def h(carry,x):
        kernel = rotspectrum(x, 0.0, vsini, 0.0, 0.0, 3.0)
        Fr=np.sum(F*kernel)/np.sum(kernel)
        return carry,Fr

    car,Fr=scan(h,0.0,dvmat)
#    Fr=[]
#    for i in range(0,M):
#        dvec=dvmat[i,:]
#        kernel = rotspectrum(dvec, 0.0, vsini, 0.0, 0.0, 3.0)
#        Fr.append(np.sum(F*kernel)/np.sum(kernel))

    Fr=np.array(Fr)
    plt.plot(wav,F[::-1],label="input",alpha=0.75)
    plt.plot(wavd,Fr[::-1],label="masuda",alpha=0.75)
    ###########################
    from PyAstronomy import pyasl

    rF = pyasl.rotBroad(wav, F[::-1], 0.0, vsini)[::-1]    
#    varr=3*10**5*jnp.log(jnp.median(wav)/wav) #km/s
    
#    plt.plot(wav,F[::-1])
    plt.plot(wav,rF[::-1],label="PyAstronomy (beta=0)",ls="dashed",alpha=0.75)
    plt.legend()
    plt.show()
    

