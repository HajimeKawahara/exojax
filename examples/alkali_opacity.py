"""Compute Na/K opacity from gf1100.all and gf1900.all in the working directory.

Files: http://kurucz.harvard.edu/linelists/gfall/gf1100.all (Na I)
       http://kurucz.harvard.edu/linelists/gfall/gf1900.all (K I)
"""

import jax
import numpy as np

from exojax.database import AdbKurucz
from exojax.opacity import OpaDirect

jax.config.update("jax_enable_x64", True)
nu_grid = np.linspace(10000.0, 50000.0, 512)  # vacuum cm-1; coarse example grid
temperatures = np.array([1000.0, 1500.0])  # K
pressures = np.array([0.01, 0.1])  # bar
for species, path in (("Na", "gf1100.all"), ("K", "gf1900.all")):
    adb = AdbKurucz(
        path, nurange=nu_grid, margin=9000.0,  # include lines whose wings reach the grid
        vmr_fraction=[0.0, 0.16, 0.84],  # H, He, H2 broadener fractions
    )
    opa = OpaDirect(adb, nu_grid, line_profile="alkali_subvoigt")
    xs = opa.xsvector(1200.0, 0.1)  # cm2 per atom
    xs_layers = opa.xsmatrix(temperatures, pressures)  # (Nlayer, Nwavenumber)
    print(species, xs.shape, xs_layers.shape)
