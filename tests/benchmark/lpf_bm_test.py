import pytest
import time 
from exojax.spec.lpf import xsvector
from exojax.spec.make_numatrix import make_numatrix0
import numpy as np

def xs(Nline=10000):
    nu0=2000.0
    nu1=2040.0
    nus=np.linspace(nu0,nu1,4000,dtype=np.float64)
    nu_lines=np.random.rand(Nline)*(nu1-nu0)+nu0
    sigmaD=np.random.rand(Nline)
    gammaL=np.random.rand(Nline)
    Sij=np.random.rand(Nline)
    
    numatrix=make_numatrix0(nus,nu_lines,warning=False)
    xsv = xsvector(numatrix,sigmaD,gammaL,Sij)
    xsv.block_until_ready()
    return True


def test_benchmark_a(benchmark):
    ret = benchmark(xs,10)
    assert ret

def test_benchmark_b(benchmark):
    ret = benchmark(xs,100)
    assert ret

def test_benchmark_c(benchmark):
    ret = benchmark(xs,1000)
    assert ret

def test_benchmark_d(benchmark):
    ret = benchmark(xs,10000)
    assert ret

def test_benchmark_e(benchmark):
    ret = benchmark(xs,100000)
    assert ret

#def test_benchmark_f(benchmark):
#    ret = benchmark(xs,1000000)
#    assert ret
