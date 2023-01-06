from exojax.spec.optgrid import optelower
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
import numpy as np
Nx = 15000
nu_grid, wav, res = wavenumber_grid(15300.0,
                                        15400.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")
Tmax = 900.0  #K
Pmin = 0.01  #bar
mdb = MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nu_grid)
print(np.max(mdb.elower))
Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
print(Eopt)

