""" 
This example determine the optimal Elower maximum for water (ExoMol, POKAZATEL), within 1 % accuracy.
We need to assume a max T (and min P) to compare the cross section whose E above Emax is cutted with the ground truth.
"""

from exojax.opacity.premodit.optgrid import optelower
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol
import numpy as np

Nx = 15000
nu_grid, wav, res = wavenumber_grid(15300.0,
                                    15400.0,
                                    Nx,
                                    unit='AA',
                                    xsmode="premodit")
Tmax = 900.0  #K
Pmin = 0.01  #bar
mdb = MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nu_grid, elower_max=20000.0)
print("Elower max in mdb = ", np.max(mdb.elower), "cm-1")
Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
print("optimal Elower max = ", Eopt, "cm-1")
