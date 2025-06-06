# correlated k distribution tests

from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
import numpy as np
from jax import config 
config.update("jax_enable_x64", True)  # use double precision

N = 70000
nus, wav, res = wavenumber_grid(6400.0, 6800.0, N, unit="cm-1", xsmode = "premodit")
print("resolution = ", res)
mdb = MdbExomol(".databases/H2O/1H2-16O/POKAZATEL/",nus)
opa = OpaPremodit(mdb, nus, auto_trange=[1000.0,1100.0])

T=1050.0
P=0.1
xsv = opa.xsvector(T, P)

import matplotlib.pyplot as plt
plt.plot(wav, xsv)
plt.savefig("corrk_test.png")
plt.close()