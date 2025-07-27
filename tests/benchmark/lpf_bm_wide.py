"""Benchmark for lpf (delta nu = 100 cm-1)
"""
import jax.numpy as jnp
import pytest
import time
from exojax.opacity.lpf.lpf import xsvector
from exojax.opacity.lpf.make_numatrix import make_numatrix0
import numpy as np


def xs(Nc, Nline=10000):

    #test1 (gpu.dat)
    # nu0=2000.0
    # nu1=2100.0
    # nus=np.linspace(nu0,nu1,10000,dtype=np.float64)

    #test2 (gpu2.dat)
    nu0 = 2000.0
    nu1 = 3000.0
    nus = np.linspace(nu0, nu1, 100000, dtype=np.float64)

    nu_lines = np.random.rand(Nline)*(nu1-nu0)+nu0
    sigmaD = np.random.rand(Nline)
    gammaL = np.random.rand(Nline)
    Sij = np.random.rand(Nline)

    numatrix = make_numatrix0(nus, nu_lines, warning=False)
    Sij_gpu = jnp.array(Sij)
    sigmaD_gpu = jnp.array(sigmaD)
    gammaL_gpu = jnp.array(gammaL)

    ts = time.time()
    a = []
    for i in range(0, Nc):
        tsx = time.time()
        #xsv = xsvector(numatrix,sigmaD,gammaL,Sij)
        xsv = xsvector(numatrix, sigmaD_gpu, gammaL_gpu, Sij_gpu)
        xsv.block_until_ready()
        tex = time.time()
        a.append(tex-tsx)
    te = time.time()
    a = np.array(a)
#    print(a)
    print(Nline, ',', np.mean(a[1:]), ',', np.std(a[1:]))

    return (te-ts)/Nc


if __name__ == '__main__':

    print('N,t_s,std_s')
    Nc = 10000
    xs(Nc, 10)

    Nc = 10000
    xs(Nc, 100)

    Nc = 1000
    xs(Nc, 1000)

    Nc = 100
    xs(Nc, 10000)

#    Nc=10
#    xs(Nc,100000)
