import pytest
import time 
from exojax.spec.redit import xsvector
from exojax.spec.dit import make_dLarray,set_ditgrid
import numpy as np
import jax.numpy as jnp
from exojax.spec import initspec

def xs(Nc,Nline=10000):
    nu0=2000.0
    nu1=2100.0
    nus=np.logspace(np.log10(nu0),np.log10(nu1),10000,dtype=np.float64)
    nu_lines=np.random.rand(Nline)*(nu1-nu0)+nu0
    nsigmaD=1.0
    gammaL=np.random.rand(Nline)+0.1

    cnu,indexnu,R, dq=initspec.init_redit(nu_lines,nus)
    ngammaL=gammaL/(nu_lines/R)
    ngammaL_grid=set_ditgrid(ngammaL,res=0.1)
    Nq=int(len(nus)/2.0)-1
    qvector=jnp.arange(-Nq,Nq+1,1)*dq   
    S=jnp.array(np.random.normal(size=Nline))
    
    ts=time.time()
    a=[]
    for i in range(0,Nc):
        tsx=time.time()
        xsv=xsvector(cnu,indexnu,R,nsigmaD,ngammaL,S,nus,ngammaL_grid,qvector)
        xsv.block_until_ready()
        tex=time.time()
        a.append(tex-tsx)
    te=time.time()
    a=np.array(a)
    print(Nline,",",np.mean(a[1:]),",",np.std(a[1:]))


    return (te-ts)/Nc

if __name__ == "__main__":

    print("N,t_s,std_s")
    Nc=100
    xs(Nc,10)

    Nc=100
    xs(Nc,100)
    
    Nc=100
    xs(Nc,1000)

    Nc=100
    xs(Nc,10000)
    
    Nc=100
    xs(Nc,100000)

#    Nc=100
#    xs(Nc,1000000)
    
