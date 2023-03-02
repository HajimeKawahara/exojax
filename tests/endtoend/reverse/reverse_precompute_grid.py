""" Reverse modeling of Methane emission spectrum using PreMODIT, precomputation of F0 grids
"""
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

from jax import random
import jax.numpy as jnp
from jax import vmap

import pandas as pd
import pkg_resources

from exojax.utils.grids import wavenumber_grid

from exojax.spec.atmrt import ArtEmisPure
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.contdb import CdbCIA
from exojax.spec.opacont import OpaCIA
from exojax.spec.response import ipgauss_sampling
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from exojax.utils.astrofunc import gravity_jupiter

from exojax.spec import molinfo
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW

filename = pkg_resources.resource_filename(
    'exojax', 'data/testdata/' + SAMPLE_SPECTRA_CH4_NEW)
dat = pd.read_csv(filename, delimiter=",", names=("wav", "flux"))
wavd = dat['wav'].values
flux = dat['flux'].values
nusd = jnp.array(1.e8 / wavd[::-1])
sigmain = 0.05
norm = 20000
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

Nx = 7500
nu_grid, wav, res = wavenumber_grid(np.min(wavd) - 10.0,
                                    np.max(wavd) + 10.0,
                                    Nx,
                                    unit='AA',
                                    xsmode='premodit')

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
diffmode = 1
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

#given gravity, temperature exponent, MMR
g = gravity_jupiter(0.88,33.2)
alpha = 0.1 
MMR_CH4 = 0.0059

# raw spectrum model given T0
def f0model(T0):
    #T-P model
    Tarr = art.powerlaw_temperature(T0, alpha)

    #molecule
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    mmr_arr = art.constant_mmr_profile(MMR_CH4)
    dtaumCH4 = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass, g)

    #continuum
    logacia_matrix = opcia.logacia_matrix(Tarr)
    dtaucH2H2 = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2,
                                        mmw, g)
    
    dtau = dtaumCH4 + dtaucH2H2
    F0 = art.run(dtau, Tarr) / norm
    #Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
    #mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV)
    return F0

# compute F0 grid given T0 grid
Ngrid = 200 # delta T = 1 K 
T0_grid = jnp.linspace(1200,1400,Ngrid) 
import tqdm
F0_grid = []
for T0 in tqdm.tqdm(T0_grid, desc="computing grid"):
    F0 = f0model(T0)
    F0_grid.append(F0)
F0_grid = jnp.array(F0_grid).T

vmapinterp = vmap(jnp.interp, (None,None,0))


print(np.shape(T0_grid),np.shape(F0_grid))
import matplotlib.pyplot as plt
plt.plot(nu_grid, vmapinterp(1295.0, T0_grid, F0_grid))
plt.plot(nusd[::-1], nflux, '+', color='black', label='data')
plt.yscale("log")
plt.show()
#import sys
#sys.exit()

def model_c(nu1, y1):
    A = numpyro.sample('A', dist.Uniform(0.1, 10.0))
    RV = numpyro.sample('RV', dist.Uniform(5.0, 15.0))
    T0 = numpyro.sample('T0', dist.Uniform(800.0, 1200.0))
    vsini = numpyro.sample('vsini', dist.Uniform(15.0, 25.0))
    F0 = A * vmapinterp(T0, T0_grid, F0_grid)
    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1=0.0, u2=0.0)
    mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV)
    numpyro.sample('y1', dist.Normal(mu, sigmain), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 300, 600
#kernel = NUTS(model_c, forward_mode_differentiation=True)
kernel = NUTS(model_c, forward_mode_differentiation=False)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, nu1=nusd, y1=nflux)
mcmc.print_summary()

# SAMPLING
posterior_sample = mcmc.get_samples()
pred = Predictive(model_c, posterior_sample, return_sites=['y1'])
predictions = pred(rng_key_, nu1=nusd, y1=None)
median_mu1 = jnp.median(predictions['y1'], axis=0)
hpdi_mu1 = hpdi(predictions['y1'], 0.9)

# PLOT
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6.0))
ax.plot(wavd[::-1], median_mu1, color='C0')
ax.plot(wavd[::-1], nflux, '+', color='black', label='data')
ax.fill_between(wavd[::-1],
                hpdi_mu1[0],
                hpdi_mu1[1],
                alpha=0.3,
                interpolate=True,
                color='C0',
                label='90% area')
plt.xlabel('wavelength ($\AA$)', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("pred_diffmode" + str(diffmode) + ".png")
plt.close()

pararr = ['A','T0', 'vsini', 'RV']
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True)
plt.savefig("corner_diffmode" + str(diffmode) + ".png")
#plt.show()
