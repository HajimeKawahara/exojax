import pytest
import time 
from exojax.spec.modit import xsvector
from exojax.spec.dit import make_dLarray,set_ditgrid
import numpy as np

def xs(Nline=10000):
    nu0=2000.0
    nu1=2040.0
    nus=np.logspace(np.log10(nu0),np.log10(nu1),4000,dtype=np.float64)
    nu_lines=np.random.rand(Nline)*(nu1-nu0)+nu0
    nsigmaD=1.0
    gammaL=np.random.rand(Nline)+0.1

    R=(len(nus)-1)/np.log(nus[-1]/nus[0]) #resolution
    dv_lines=nu_lines/R
    dv=nus/R
    Nfold=2
    dLarray=make_dLarray(Nfold,1.0)
    ngammaL=gammaL/dv_lines
    ngammaL_grid=set_ditgrid(ngammaL,res=0.1)
    S=np.random.normal(size=Nline)
    xsv=xsvector(nu_lines,nsigmaD,ngammaL,S,nus,ngammaL_grid,dLarray,dv_lines,dv)
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

#def test_benchmark_g(benchmark):
#    ret = benchmark(xs,10000000)
#    assert ret

    
#if __name__ == "__main__":
#    xs(Nline=10000)
