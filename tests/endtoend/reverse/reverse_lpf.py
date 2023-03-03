#!/usr/bin/env python
import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random
from exojax.spec import contdb
from exojax.spec.api import MdbExomol
from exojax.spec import molinfo
from exojax.spec.response import ipgauss_sampling_slow
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from exojax.spec.rtransfer import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec.opacalc import OpaDirect
from exojax.spec.opacont import OpaCIA
from exojax.spec.atmrt import ArtEmisPure
from exojax.spec.unitconvert import wav2nu
from exojax.utils.astrofunc import gravity_jupiter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

dat = pd.read_csv('spectrum.txt', delimiter=',', names=('wav', 'flux'))
wavd = dat['wav'].values
flux = dat['flux'].values
nusd = wav2nu(wavd)
sigmain = 0.05
norm = 40000.
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

nu_grid, wav, res = wavenumber_grid(np.min(wavd) - 5.0,
                                    np.max(wavd) + 5.0,
                                    1500,
                                    unit='AA')

art = ArtEmisPure(nu_grid, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)
art.change_temperature_range(400.0, 1500.0)

instrumental_resolution = 100000.
beta_inst = resolution_to_gaussian_std(instrumental_resolution)
Mp = 33.2  # fixing mass...

mdbCO = MdbExomol('.database/CO/12C-16O/Li2015',
                  nu_grid,
                  crit=1.e-46,
                  gpu_transfer=True)
opa = OpaDirect(mdb=mdbCO, nu_grid=nu_grid)
cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nu_grid)
opacia = OpaCIA(cdbH2H2, nu_grid)
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass_isotope('H2')
vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR

#settings before HMC
vsini_max = 100.0
vr_array = velocity_grid(res, vsini_max)


def model_c(nu1, y1):
    Rp = numpyro.sample('Rp', dist.Uniform(0.4, 1.2))
    RV = numpyro.sample('RV', dist.Uniform(5.0, 15.0))
    MMR_CO = numpyro.sample('MMR_CO', dist.Uniform(0.0, 0.015))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0, 1500.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05, 0.2))
    vsini = numpyro.sample('vsini', dist.Uniform(15.0, 25.0))
    gravity = gravity_jupiter(Mp, Rp)  # gravity in the unit of Jupiter
    u1 = 0.0
    u2 = 0.0

    # T-P model//
    Tarr = art.powerlaw_temperature(T0, alpha)
    mmr_arr = art.constant_mmr_profile(MMR_CO)

    def obyo(y, tag, nusd, nus):
        # CO
        xsm_CO = opa.xsmatrix(Tarr, art.pressure)
        dtaumCO = art.opacity_profile_lines(xsm_CO, mmr_arr, opa.mdb.molmass,
                                            gravity)
        # CIA
        logacia = opacia.logacia_matrix(Tarr)
        dtaucH2H2 = art.opacity_profile_cia(logacia, Tarr, vmrH2, vmrH2, mmw,
                                            gravity)
        dtau = dtaumCO + dtaucH2H2
        F0 = art.run(dtau, Tarr) / norm
        Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
        mu = ipgauss_sampling_slow(nusd, nus, Frot, beta_inst, RV)
        numpyro.sample(tag, dist.Normal(mu, sigmain), obs=y)

    obyo(y1, 'y1', nu1, nu_grid)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 300, 600
#kernel = NUTS(model_c, forward_mode_differentiation=True)
kernel = NUTS(model_c, forward_mode_differentiation=False)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, nu1=nusd, y1=nflux)

posterior_sample = mcmc.get_samples()
pred = Predictive(model_c, posterior_sample, return_sites=['y1'])
predictions = pred(rng_key_, nu1=nusd, y1=None)
median_mu1 = jnp.median(predictions['y1'], axis=0)
hpdi_mu1 = hpdi(predictions['y1'], 0.9)
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
plt.savefig("spectrum.png")
plt.close()

pararr = ['Rp', 'T0', 'alpha', 'MMR_CO', 'vsini', 'RV']
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True)
plt.savefig("corner.png")
plt.close()
