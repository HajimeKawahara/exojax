import matplotlib.pyplot as plt
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit

from jax.config import config
config.update("jax_enable_x64", True)

nu_grid,wav,res=wavenumber_grid(1900.0,2300.0,200000,xsmode="premodit",unit="cm-1",)
mdb = MdbExomol(".database/CO/12C-16O/Li2015",nu_grid)
opa = OpaPremodit(mdb,nu_grid,auto_trange=[950.0,1050.0])
xsv = opa.xsvector(1000.0, 1.0)

plt.plot(nu_grid, xsv)
plt.yscale('log')
plt.show()
