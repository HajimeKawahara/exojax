# correlated k distribution tests

from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
import numpy as np

import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
import matplotlib.pyplot as plt
from exojax.spec.opacalc import OpaPremodit
from jax import config 
config.update("jax_enable_x64", True)  # use double precision

fig = False

N = 70000
#nus, wav, res = wavenumber_grid(6400.0, 6800.0, N, unit="cm-1", xsmode = "premodit")
#mdb = MdbExomol(".databases/H2O/1H2-16O/POKAZATEL/",nus)
#print("resolution = ", res)

nus, wav, res = mock_wavenumber_grid(lambda0=22920.0, lambda1=22940.0, Nx=2000)
mdb = mock_mdbExomol("H2O")

T=1000.0
P = 1.e-2
opa = OpaPremodit(mdb, nus, auto_trange=[1000.0,1100.0])

xsv = opa.xsvector(T,P)
indx = jnp.argsort(xsv)

plt.plot(xsv[indx], label="xsv sorted")
plt.yscale("log")
plt.xlabel("Index")
plt.ylabel("Cross section (cm2/molecule)")
plt.show()


if fig:
    plt.plot(nus, xsv, label="xsv")
    plt.yscale("log")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Cross section (cm2/molecule)")
    plt.show()
