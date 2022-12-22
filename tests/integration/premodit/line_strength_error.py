""" This test accounts for Issue #288, bug fix large elower value using f32, 
    The bug was due to the overflow in the function when computing,
"""

from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from exojax.utils.grids import wavenumber_grid
from exojax.spec import api
from exojax.spec import initspec, molinfo, premodit
from exojax.spec import molinfo
from exojax.utils.constants import RJ, pc
from exojax.spec.premodit import unbiased_lsd
from exojax.spec.lsd import inc2D_givenx
from exojax.spec import modit
from exojax.spec.hitran import line_strength
from exojax.utils.grids import wavenumber_grid
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.spec.exomol import gamma_exomol

wls, wll, Ndata = 15035, 15040, 100
wavd = np.linspace(wls, wll, Ndata)
nusd = jnp.array(1.e8 / wavd[::-1])

Nx = 2000
nus, wav, reso = wavenumber_grid(np.min(wavd) - 5.0,
                                 np.max(wavd) + 5.0,
                                 Nx,
                                 unit="AA",
                                 xsmode="modit")

#Load H2O data with premodit
molmassH2O = molinfo.molmass_major_isotope("H2O")
mdb = api.MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nus, gpu_transfer=True)
print('N_H2O=', len(mdb.nu_lines))

#Pre MODIT LSD
Tgue = 1000.
interval_contrast = 0.3
dit_grid_resolution = 0.1
lbd, multi_index_uniqgrid_H2O, elower_grid, \
ngamma_ref_grid_H2O, n_Texp_grid_H2O, R_H2O, pmarray_H2O = initspec.init_premodit(
    mdb.nu_lines,
    nus,
    mdb.elower,
    mdb.alpha_ref,
    mdb.n_Texp,
    mdb.Sij0,
    Ttyp=Tgue,
    interval_contrast=interval_contrast,
    dit_grid_resolution=dit_grid_resolution,
    warning=False)

print("Elower = ",np.max(mdb.elower),"-",np.min(mdb.elower),"cm-1")
dE=elower_grid[1]-elower_grid[0]
NE=len(elower_grid)
print("dE = ",dE,"cm-1")
print("NEgrid = ",NE)

T = 700.0
P = 1.0
qt = mdb.qr_interp(T)
Slsd_premodit = unbiased_lsd(lbd, T, nus, elower_grid, qt)
Spremodit = (np.sum(Slsd_premodit,axis=1))

#MODIT LSD
cont, index, R, pmarray = initspec.init_modit(
    mdb.nu_lines, nus)
Sij = line_strength(T, mdb.logsij0, mdb.nu_lines, mdb.elower, qt)
gammaL = gamma_exomol(P, T, mdb.n_Texp, mdb.alpha_ref)
dv_lines = mdb.nu_lines / R
ngammaL = gammaL / dv_lines
ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)    
log_ngammaL_grid = jnp.log(ngammaL_grid)
lsd_array = jnp.zeros((len(nus), len(ngammaL_grid)))
Slsd_modit = inc2D_givenx(lsd_array, Sij, cont, index, jnp.log(ngammaL),
                        log_ngammaL_grid)
Smodit = (np.sum(Slsd_modit,axis=1))


import matplotlib.pyplot as plt
fig = plt.figure()
#ax = fig.add_subplot(211)
#plt.plot(mdb.nu_lines,mdb.elower,".",alpha=0.1)
#plt.yscale("log")
#ax = fig.add_subplot(212)
plt.plot(nus,Spremodit/Smodit-1.0)
#plt.yscale("log")
plt.xlabel("wavenumber cm-1")
plt.ylabel("relative error from MODIT LSD")
plt.title("H2O/POK ic="+str(interval_contrast)+" dE="+str(int(dE))+"cm-1 Tguess="+
          str(int(Tgue))+"K T="+str(int(T))+"K NE="+str(NE))
plt.savefig("ic"+str(interval_contrast)+"_"+str(int(T))+"K.png")
plt.show()
