""" This code compares Premodit line strengths with those of MODIT for Hitemp.
    This test accounts for Issue #288, bug fix large elower value using f32, 
    The bug was due to the overflow in the function when computing.
    We also provide an example of manual calculation of cross section using PreMODIT as well as using opa.xsvector.
"""
import numpy as np
import jax.numpy as jnp
from exojax.utils.grids import wavenumber_grid
from exojax.spec import api
from exojax.spec.opacalc import OpaPremodit
from exojax.spec import initspec
from exojax.spec.premodit import unbiased_lsd_zeroth
from exojax.spec.premodit import unbiased_lsd_first
from exojax.spec.premodit import unbiased_lsd_second
from exojax.spec.lsd import inc2D_givenx
from exojax.spec.hitran import line_strength
from exojax.utils.grids import wavenumber_grid
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.spec.hitran import gamma_hitran
from exojax.spec.hitran import gamma_natural
from exojax.utils.constants import Tref_original
from exojax.test.emulate_mdb import mock_mdbHitemp
## also, xs
from exojax.spec import normalized_doppler_sigma
from exojax.spec.modit_scanfft import calc_xsection_from_lsd_scanfft
from exojax.spec.premodit import unbiased_ngamma_grid

from jax import config
config.update("jax_enable_x64", True)

Nx = 5000
nus, wav, res = wavenumber_grid(22800.0,
                                23100.0,
                                Nx,
                                unit='AA',
                                xsmode="modit")

mdb = api.MdbHitemp('CO', nus, gpu_transfer=True, isotope=1)

diffmode = 0

Ttest = 1200.0
P = 1.0

#PreMODIT LSD
opa = OpaPremodit(mdb=mdb,
                  nu_grid=nus,
                  auto_trange=[1000.0, 1500.0],
                  diffmode=diffmode)
lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo

#automatic computing by xsvector
xsv = opa.xsvector(Ttest,P)

#tries manual computation of xsvector below
qt = mdb.qr_interp(mdb.isotope, Ttest)
dE = opa.dE
NE = len(elower_grid)

if diffmode == 0:
    Slsd_premodit = unbiased_lsd_zeroth(lbd_coeff[0], Ttest, opa.Tref, nus,
                                        elower_grid, qt)
elif diffmode == 1:
    Slsd_premodit = unbiased_lsd_first(lbd_coeff, Ttest, opa.Tref, opa.Twt,
                                       opa.nu_grid, elower_grid, qt)
elif diffmode == 2:
    Slsd_premodit = unbiased_lsd_second(lbd_coeff, Ttest, opa.Tref, opa.Twt,
                                        opa.nu_grid, elower_grid, qt)

Spremodit = (np.sum(Slsd_premodit, axis=1))
nsigmaD = normalized_doppler_sigma(Ttest, mdb.molmass, R)
ngamma_grid = unbiased_ngamma_grid(Ttest, P, ngamma_ref_grid, n_Texp_grid,
                                   multi_index_uniqgrid, opa.Tref_broadening)
log_ngammaL_grid = jnp.log(ngamma_grid)
xsv_manual = calc_xsection_from_lsd_scanfft(Slsd_premodit, R, pmarray,
                                             nsigmaD, nus, log_ngammaL_grid)
#===========================================================================
# MODIT LSD
# We need to revert the reference temperature to 296K to reuse mdb for MODIT
#===========================================================================
from exojax.spec.modit import xsvector
from exojax.spec.initspec import init_modit

mdb.change_reference_temperature(Tref_original)
qt = mdb.qr_interp(mdb.isotope, Ttest)
cont, index, R, pmarray = initspec.init_modit(mdb.nu_lines, nus)
Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
gammaL = gamma_hitran(P, Ttest, 0.0, mdb.n_air, mdb.gamma_air,
                      mdb.gamma_self) + gamma_natural(mdb.A)

dv_lines = mdb.nu_lines / R
ngammaL = gammaL / dv_lines
ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
log_ngammaL_grid = jnp.log(ngammaL_grid)
lsd_array = jnp.zeros((len(nus), len(ngammaL_grid)))
Slsd_modit = inc2D_givenx(lsd_array, Sij, cont, index, jnp.log(ngammaL),
                          log_ngammaL_grid)
Smodit = (np.sum(Slsd_modit, axis=1))

## also, xs
Sij = line_strength(Ttest, mdb.logsij0, mdb.nu_lines, mdb.elower, qt, mdb.Tref)
cont_nu, index_nu, R, pmarray = init_modit(mdb.nu_lines, nus)
ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
xsv_modit = xsvector(cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus,
                     ngammaL_grid)
#xsv_modit_sld = xsvector(cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Smodit, nus,
#                     ngammaL_grid)

from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_XS_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_XS_REF_AIR
import pkg_resources
import pandas as pd

filename = pkg_resources.resource_filename(
    'exojax', 'data/testdata/' + TESTDATA_CO_HITEMP_MODIT_XS_REF_AIR)
dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))

#np.savetxt("xsv_modit.txt", np.array([nus, xsv_modit]).T, delimiter=",")

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(nus, xsv, label="premodit", ls="dashed")
plt.plot(nus, xsv_manual, label="premodit (manual)", ls="dashed")
plt.plot(nus, xsv_modit, label="modit", ls="dotted")
plt.yscale("log")
plt.legend()
ax = fig.add_subplot(212)
plt.plot(nus, xsv / xsv_modit - 1.0, label="premodit", ls="dashed")
plt.plot(nus, xsv_manual / xsv_modit - 1.0, label="premodit (manual)", ls="dashed")

ax.set_ylim(-0.03, 0.03)
ax.axhline(0.01, color="gray", ls="dashed")
ax.axhline(-0.01, color="gray", ls="dashed")
ax.axhline(0.0, color="gray")
plt.show()

mask = Spremodit > 0.0

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(nus, Spremodit, ".")
plt.xscale("log")
plt.yscale("log")
ax = fig.add_subplot(212)
plt.plot(nus[mask], Spremodit[mask] / Smodit[mask] - 1.0)
#plt.yscale("log")
plt.xlabel("wavenumber cm-1")
plt.ylabel("relative error from MODIT LSD")
plt.title("CO dE=" + str(int(dE)) + "cm-1, T=" + str(int(Ttest)) + "K NE=" +
          str(NE))
#plt.savefig("dE" + str(dE) + "_" + str(int(T)) + "K.png")
plt.show()
