""" This code compares Premodit line strengths with those of MODIT for ExoMol.
    This test accounts for Issue #288, bug fix large elower value using f32, 
    The bug was due to the overflow in the function when computing,
"""

from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol 
from exojax.opacity import OpaPremodit
from exojax.opacity import initspec, molinfo
from exojax.database import molinfo 
from exojax.opacity.premodit.premodit import unbiased_lsd_zeroth
from exojax.opacity.premodit.premodit import unbiased_lsd_first
from exojax.opacity.premodit.premodit import unbiased_lsd_second
from exojax.opacity._common.lsd import inc2D_givenx
from exojax.opacity.modit import modit
from exojax.database.core.line_strength import line_strength
from exojax.utils.grids import wavenumber_grid
from exojax.opacity._common.set_ditgrid import ditgrid_log_interval
from exojax.database.core.broadening  import gamma_exomol
from exojax.utils.constants import Tref_original

wls, wll, Ndata = 15035, 15040, 100
wavd = np.linspace(wls, wll, Ndata)
nusd = jnp.array(1.e8 / wavd[::-1])

Nx = 2000
wavmin = np.min(wavd) - 5.0
wavmax = np.max(wavd) + 5.0
nus, wav, reso = wavenumber_grid(wavmin, wavmax, Nx, unit="AA", xsmode="modit")
#Load H2O data with premodit
molmassH2O = molinfo.molmass_isotope("H2O")
mdb =MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nus, gpu_transfer=True)
#mdb.change_reference_temperature(Tref)
print('N_H2O=', len(mdb.nu_lines))

diffmode = 2

T = 1200.0
P = 1.0

#Pre MODIT LSD
dit_grid_resolution = 0.1
opa = OpaPremodit(mdb=mdb, nu_grid=nus, auto_trange=[1000.0,1200.0],diffmode=diffmode)
lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    
qt = mdb.qr_interp(T, opa.Tref)
dE = opa.dE
NE = len(elower_grid)

if diffmode == 0:
    Slsd_premodit = unbiased_lsd_zeroth(lbd_coeff[0], T, opa.Tref, nus, elower_grid, qt)
elif diffmode == 1:
    Slsd_premodit = unbiased_lsd_first(lbd_coeff, T, opa.Tref, opa.Twt, opa.nu_grid, elower_grid, qt)
elif diffmode == 2:
    Slsd_premodit = unbiased_lsd_second(lbd_coeff, T, opa.Tref, opa.Twt, opa.nu_grid, elower_grid, qt)

Spremodit = (np.sum(Slsd_premodit, axis=1))

#===========================================================================
# MODIT LSD
# We need to revert the reference temperature to 296K to reuse mdb for MODIT
#===========================================================================

qt = mdb.qr_interp(T, Tref_original)
cont, index, R, pmarray = initspec.init_modit(mdb.nu_lines, nus)
Sij = line_strength(T, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, Tref_original)
gammaL = gamma_exomol(P, T, mdb.n_Texp, mdb.alpha_ref)
dv_lines = mdb.nu_lines / R
ngammaL = gammaL / dv_lines
ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
log_ngammaL_grid = jnp.log(ngammaL_grid)
lsd_array = jnp.zeros((len(nus), len(ngammaL_grid)))
Slsd_modit = inc2D_givenx(lsd_array, Sij, cont, index, jnp.log(ngammaL),
                          log_ngammaL_grid)
Smodit = (np.sum(Slsd_modit, axis=1))

import matplotlib.pyplot as plt

fig = plt.figure()
#ax = fig.add_subplot(211)
#plt.plot(mdb.nu_lines,mdb.elower,".",alpha=0.1)
#plt.yscale("log")
#ax = fig.add_subplot(212)
plt.plot(nus, Spremodit / Smodit - 1.0)
#plt.yscale("log")
plt.xlabel("wavenumber cm-1")
plt.ylabel("relative error from MODIT LSD")
plt.title("H2O/POK dE=" + str(int(dE)) + "cm-1, T=" + str(int(T)) + "K NE=" +
          str(NE))
#plt.savefig("dE" + str(dE) + "_" + str(int(T)) + "K.png")
plt.show()
