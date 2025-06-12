"""This module tests the correlated k distribution implementation in Exojax."""

import numpy as np
import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
import matplotlib.pyplot as plt
from exojax.opacity.opacalc import OpaPremodit
from jax import config

config.update("jax_enable_x64", True)  # use double precision

# from exojax.utils.grids import wavenumber_grid
# from exojax.database.api  import MdbExomol
# N = 70000
# nus, wav, res = wavenumber_grid(6400.0, 6800.0, N, unit="cm-1", xsmode = "premodit")
# mdb = MdbExomol(".databases/H2O/1H2-16O/POKAZATEL/",nus)
# print("resolution = ", res)

nus, wav, res = mock_wavenumber_grid(lambda0=22930.0, lambda1=22940.0, Nx=20000)
mdb = mock_mdbExomol("H2O")

def compute_g(xsv):
    idx = jnp.argsort(xsv)
    k_g = xsv[idx]
    g = jnp.arange(xsv.size, dtype=xsv.dtype) / xsv.size
    return idx, k_g, g


def gauss_legendre(Ng):
    x, w = np.polynomial.legendre.leggauss(Ng)  # [-1,1]
    gpoint = 0.5 * (1.0 + x)
    return gpoint, 0.5 * w


Ng = 32
ggrid, weights = gauss_legendre(Ng)


opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])

T = 1000.0
P = 1.0e-2
xsv = opa.xsvector(T, P)
idx, k_g, g = compute_g(xsv)
log_k_g = jnp.log(k_g)  # log(k_g) for interpolation

from jax.numpy import interp
log_kggrid = interp(ggrid, g, log_k_g)  # interpolate ggrid to the sorted xsv


#dnus_ = nus[1] - nus[0] 
dnus_ = nus[-1] - nus[-2] 
L = 1.e22
print(np.sum(jnp.exp(-xsv*L)*dnus_))

dnus_whole = nus[-1] - nus[0]  # the whole range of nus
print(jnp.sum(weights*jnp.exp(-jnp.exp(log_kggrid)*L))*dnus_whole)

print(res)

exit()
