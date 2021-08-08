""" benchmark of lpf for 


"""

import pytest
import time 
from exojax.spec.lpf import xsvector
from exojax.spec.make_numatrix import make_numatrix0
import numpy as np

def xsm(Nline):
    Nj=int(Nline/10000)
    arr=np.array([range(0,Nj),range(1,Nj+1)],dtype=np.int).T*10000
    print(arr)
    nu0=2000.0
    nu1=2100.0
    nus=np.linspace(nu0,nu1,10000,dtype=np.float64)
    nu_lines=np.random.rand(Nline)*(nu1-nu0)+nu0
    sigmaD=np.random.rand(Nline)
    gammaL=np.random.rand(Nline)
    Sij=np.random.rand(Nline)
    for ia in arr:
        i0=ia[0]
        i1=ia[1]
        numatrix=make_numatrix0(nus,nu_lines[i0:i1],warning=False)
        xsvtmp = xsvector(numatrix,sigmaD[i0:i1],gammaL[i0:i1],Sij[i0:i1])
        del numatrix
        
    return True


def test_benchmark_A(benchmark):
    ret = benchmark(xsm,50000)
    assert ret

def test_benchmark_B(benchmark):
    ret = benchmark(xsm,100000)
    assert ret

def test_benchmark_C(benchmark):
    ret = benchmark(xsm,1000000)
    assert ret

    
#if __name__ == "__main__":
#    xsm(50000)
