import pytest
import time
from exojax.opacity.modit.modit import xsvector_scanfft
from exojax.opacity._common.set_ditgrid import ditgrid_log_interval
from exojax.opacity import initspec
import jax.numpy as jnp
import numpy as np


def xs(Nline):
    nu0 = 2000.0
    nu1 = 2100.0
    nus = np.logspace(np.log10(nu0), np.log10(nu1), 10000, dtype=np.float64)
    nu_lines = np.random.rand(Nline)*(nu1-nu0)+nu0
    nsigmaD = 1.0
    gammaL = np.random.rand(Nline)+0.1
    cnu, indexnu, R, pmarray = initspec.init_modit(nu_lines, nus)
    ngammaL = gammaL/(nu_lines/R)
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    S = jnp.array(np.random.normal(size=Nline))
    xsv = xsvector_scanfft(cnu, indexnu, R, pmarray, nsigmaD,
                   ngammaL, S, nus, ngammaL_grid)
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


def test_benchmark_f(benchmark):
    ret = benchmark(xs, 1000000)
    assert ret

# def test_benchmark_g(benchmark):
#    ret = benchmark(xs,10000000)
#    assert ret


# if __name__ == "__main__":
#    xs(Nline=10000)
