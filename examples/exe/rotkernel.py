import jax.numpy as jnp
from jax import jit


if __name__ == "__main__":
    from PyAstronomy import pyasl
    from exojax.spec import AutoRT
    from exojax.spec import response
    import numpy as np 
    import matplotlib.pyplot as plt
    import sys
    from jax.lax import scan

    c=299792.458
    RV=0.0
    vsini=15.0
    beta=3.0
    u1=0.0
    u2=0.0

    #grid for F0
    N=10000
    wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
    nus=1.e8/wav[::-1]

    #grid for F
    M=7500
    wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA        
    nusd=1.e8/wavd[::-1]

    #dv matrix
    dvmat=jnp.array(c*np.log(nusd[:,None]/nus[None,:]))

    
    #compute a TOA spectrum
    Parr=np.logspace(-8,2,100) #100 pressure layers (10**-8 to 100 bar)
    Tarr = 1500.*(Parr/Parr[-1])**0.02    #some T-P profile
    autort=AutoRT(nus,1.e5,Tarr,Parr)     #g=1.e5 cm/s2
    autort.addmol("ExoMol","CO",0.01)     #mmr=0.01
    F0=autort.rtrun()
    Fr=response.response(dvmat,F0,u1,u2,vsini,beta,RV)

    #reference by PyAstronomy
    rF = pyasl.rotBroad(wav, F0[::-1], 0.0, vsini)[::-1]    


    plt.plot(wav,F0[::-1],label="input (F0)",alpha=0.75)
    plt.plot(wavd,Fr[::-1],label="exojax",alpha=0.75)
    plt.plot(wav,rF[::-1],label="PyAstronomy ($\beta=0$)",ls="dashed",alpha=0.75)
    plt.legend()
    plt.show()
    

