import pytest
import time 
from exojax.spec.modit import xsvector
from exojax.spec.dit import make_dLarray,set_ditgrid
import numpy as np
import jax.numpy as jnp

def xs(Nc,Nline=10000):
    nu0=2000.0
    nu1=2100.0
    nus=np.logspace(np.log10(nu0),np.log10(nu1),10000,dtype=np.float64)
    nu_lines=np.random.rand(Nline)*(nu1-nu0)+nu0
    nsigmaD=1.0
    gammaL=np.random.rand(Nline)+0.1

    R=(len(nus)-1)/np.log(nus[-1]/nus[0]) #resolution
    dv_lines=jnp.array(nu_lines/R)
    dv=jnp.array(nus/R)
    Nfold=2
    dLarray=make_dLarray(Nfold,1.0)
    ngammaL=gammaL/dv_lines
    
    ngammaL_grid=jnp.array(set_ditgrid(ngammaL,res=0.1))
    S=jnp.array(np.random.normal(size=Nline))
    ts=time.time()
    a=[]
    for i in range(0,Nc):
        tsx=time.time()
        xsv=xsvector(nu_lines,nsigmaD,ngammaL,S,nus,ngammaL_grid,dLarray,dv_lines,dv)
        xsv.block_until_ready()
        tex=time.time()
        a.append(tex-tsx)
    te=time.time()
    a=np.array(a)
    print(Nline,",",np.mean(a[1:]),",",np.std(a[1:]))


    return (te-ts)/Nc

if __name__ == "__main__":

    print("N,t_s,std_s")
    Nc=10000
    xs(Nc,10)

    Nc=1000
    xs(Nc,100)
    
    Nc=100
    xs(Nc,1000)

    Nc=100
    xs(Nc,10000)
    
    Nc=100
    xs(Nc,100000)

    Nc=100
    xs(Nc,1000000)
    
