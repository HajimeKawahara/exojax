import pytest
import time
from exojax.opacity.modit.modit import xsvector_scanfft
from exojax.opacity._common.set_ditgrid import ditgrid_log_interval
import numpy as np
import jax.numpy as jnp
from exojax.opacity import initspec


def xs(Nc, Nline=10000):
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

    ts = time.time()
    a = []
    for i in range(0, Nc):
        tsx = time.time()
        xsv = xsvector_scanfft(cnu, indexnu, R, pmarray, nsigmaD,
                       ngammaL, S, nus, ngammaL_grid)
        xsv.block_until_ready()
        tex = time.time()
        a.append(tex-tsx)
    te = time.time()
    a = np.array(a)
    print(Nline, ',', np.mean(a[1:]), ',', np.std(a[1:]))

    return (te-ts)/Nc


if __name__ == '__main__':

    print('N,t_s,std_s')
    Nc = 10000
    xs(Nc, 10)

    Nc = 1000
    xs(Nc, 100)

    Nc = 100
    xs(Nc, 1000)

    Nc = 100
    xs(Nc, 10000)

    Nc = 100
    xs(Nc, 100000)

    Nc = 100
    xs(Nc, 1000000)

    Nc = 100
    xs(Nc, 10000000)

    Nc = 100
    xs(Nc, 100000000)
