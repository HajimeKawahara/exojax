"""Benchmark for lpf (delta nu = 100 cm-1)
"""

import pytest
import time
from exojax.opacity.lpf.lpf import xsvector
from exojax.opacity.lpf.make_numatrix import make_numatrix0
import numpy as np


def xs(Nline):
    nu0 = 2000.0
    nu1 = 2100.0
    nus = np.linspace(nu0, nu1, 10000, dtype=np.float64)

    #    nu0=2000.0
    #    nu1=3000.0
    #    nus=np.linspace(nu0,nu1,100000,dtype=np.float64)
    nu_lines = np.random.rand(Nline)*(nu1-nu0)+nu0
    sigmaD = np.random.rand(Nline)
    gammaL = np.random.rand(Nline)
    Sij = np.random.rand(Nline)

    numatrix = make_numatrix0(nus, nu_lines, warning=False)
    xsv = xsvector(numatrix, sigmaD, gammaL, Sij)
    xsv.block_until_ready()
    return True


def test_benchmark_a(benchmark):
    ret = benchmark(xs, 10)
    assert ret


def test_benchmark_b(benchmark):
    ret = benchmark(xs, 100)
    assert ret


def test_benchmark_c(benchmark):
    ret = benchmark(xs, 1000)
    assert ret


def test_benchmark_d(benchmark):
    ret = benchmark(xs, 10000)
    assert ret


def test_benchmark_e(benchmark):
    ret = benchmark(xs, 100000)
    assert ret
