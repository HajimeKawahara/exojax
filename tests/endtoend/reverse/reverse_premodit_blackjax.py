""" Reverse modeling of Methane emission spectrum using MODIT
"""
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

import pandas as pd
import pkg_resources

from exojax.spec.atmrt import ArtEmisPure
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.contdb import CdbCIA
from exojax.spec.opacont import OpaCIA
from exojax.spec.response import ipgauss_sampling
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.spec import molinfo
from exojax.spec.unitconvert import nu2wav
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW

from jax import config

config.update("jax_enable_x64", True)

# PPL
import arviz
import blackjax
import jax.scipy.stats as stats
from jax import random

# loading the data
filename = pkg_resources.resource_filename(
    'exojax', 'data/testdata/' + SAMPLE_SPECTRA_CH4_NEW)
dat = pd.read_csv(filename, delimiter=",", names=("wavenumber", "flux"))
nusd = dat['wavenumber'].values
flux = dat['flux'].values
wavd = nu2wav(nusd, wavelength_order="ascending")

sigmain = 0.05
norm = 20000
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

Nx = 7500
nu_grid, wav, res = wavenumber_grid(np.min(wavd) - 10.0,
                                    np.max(wavd) + 10.0,
                                    Nx,
                                    unit='AA',
                                    xsmode='premodit', wavelength_order="ascending")

Tlow = 400.0
Thigh = 1500.0
art = ArtEmisPure(nu_grid, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)
art.change_temperature_range(Tlow, Thigh)
Mp = 33.2

Rinst = 100000.
beta_inst = resolution_to_gaussian_std(Rinst)

### CH4 setting (PREMODIT)
mdb = MdbExomol('.database/CH4/12C-1H4/YT10to10/',
                nurange=nu_grid,
                gpu_transfer=False)
print('N=', len(mdb.nu_lines))
diffmode = 0
opa = OpaPremodit(mdb=mdb,
                  nu_grid=nu_grid,
                  diffmode=diffmode,
                  auto_trange=[Tlow, Thigh],
                  dit_grid_resolution=0.2)

## CIA setting
cdbH2H2 = CdbCIA('.database/H2-H2_2011.cia', nu_grid)
opcia = OpaCIA(cdb=cdbH2H2, nu_grid=nu_grid)
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass_isotope('H2')
vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR

#settings before HMC
vsini_max = 100.0
vr_array = velocity_grid(res, vsini_max)

print("ready")


def frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini):
    g = gravity_jupiter(Rp=Rp, Mp=Mp)  # gravity in the unit of Jupiter
    #molecule
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    mmr_arr = art.constant_mmr_profile(MMR_CH4)
    dtaumCH4 = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, g)
    #continuum
    logacia_matrix = opcia.logacia_matrix(Tarr)
    dtaucH2H2 = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2,
                                        mmw, g)
    #total tau
    dtau = dtaumCH4 + dtaucH2H2
    F0 = art.run(dtau, Tarr) / norm
    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
    mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV, vr_array)
    return mu


import matplotlib.pyplot as plt
#g = gravity_jupiter(0.88, 33.2)
Rp = 0.88
Mp = 33.2
alpha = 0.1
MMR_CH4 = 0.0059
vsini = 20.0
RV = 10.0
T0 = 1200.0
u1 = 0.0
u2 = 0.0
Tarr = art.powerlaw_temperature(T0, alpha)
Ftest = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)

Tarr = art.powerlaw_temperature(1500.0, alpha)
Ftest2 = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)

plt.plot(nusd, nflux)
plt.plot(nusd, Ftest, ls="dashed")
plt.plot(nusd, Ftest2, ls="dotted")
plt.yscale("log")
plt.show()


def g(Rp, RV, MMR_CH4, T0, alpha, vsini):
    Tarr = art.powerlaw_temperature(T0, alpha)
    return frun(Tarr, MMR_CH4, Mp, Rp, 0.0, 0.0, RV, vsini)


def logprob_fn(x):
    logpdf = stats.norm.logpdf(
        nflux,
        g(x["Rp"], x["RV"], x["MMR_CH4"], x["T0"], x["alpha"], x["vsini"]),
        1.0)
    return jnp.sum(logpdf) 


step_size = 1e-3
inverse_mass_matrix = jnp.array([0.1, 1., 0.001, 100., 0.001, 1.])
nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

# Initialize the state
initial_position = {
    "Rp": 0.5,
    "RV": 10.0,
    "MMR_CH4":0.01,
    "T0":1200.0,
    "alpha":0.12,
    "vsini":12.0
    }
state = nuts.init(initial_position)

# Iterate
rng_key = random.PRNGKey(0)

#warmup = blackjax.window_adaptation(
#    blackjax.nuts,
#    logprob_fn,
#    100,
#)

#state, kernel, _ = warmup.run(
#    rng_key,
#    res.x,
#)
#print(state)
#print(kernel)

for _ in range(100):
    _, rng_key = random.split(rng_key)
    y = nuts.step(rng_key, state)
    print(y)
    state, _ = y
