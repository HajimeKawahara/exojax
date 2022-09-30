""" Reverse modeling of Methane emission spectrum using MODIT
"""

#!/usr/bin/env python
# coding: utf-8
import arviz
from exojax.spec.modit import setdgm_exomol
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from exojax.spec import rtransfer as rt
from exojax.spec import moldb, contdb
from exojax.spec import rtransfer
from exojax.spec.rtransfer import nugrid
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
from exojax.spec import planck, response
from exojax.spec import molinfo
from exojax.utils.constants import RJ
from exojax.utils.instfunc import R2STD
import numpy as np
from exojax.spec import initspec
import pkg_resources
from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW
from exojax.spec.premodit import xsmatrix
from exojax.spec.planck import piBarr
from jax import vmap

filename = pkg_resources.resource_filename(
    'exojax', 'data/testdata/' + SAMPLE_SPECTRA_CH4_NEW)
dat = pd.read_csv(filename, delimiter=",", names=("wav", "flux"))
wavd = dat['wav'].values
flux = dat['flux'].values
nusd = jnp.array(1.e8 / wavd[::-1])
sigmain = 0.05
norm = 20000
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

NP = 100
Parr, dParr, k = rtransfer.pressure_layer(NP=NP)
Nx = 7500
nu_grid, wav, res = nugrid(np.min(wavd) - 10.0,
                           np.max(wavd) + 10.0,
                           Nx,
                           unit='AA',
                           xsmode='premodit')
Rinst = 100000.
beta_inst = R2STD(Rinst)

molmassCH4 = molinfo.molmass('CH4')
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass('H2')
vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR

#
Mp = 33.2
mdb = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nu_grid, crit=1.e-30)
cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nu_grid)
print('N=', len(mdb.nu_lines))

# Reference pressure for a T-P model
Pref = 1.0  # bar
ONEARR = np.ones_like(Parr)
ONEWAV = jnp.ones_like(nflux)

interval_contrast = 0.1
dit_grid_resolution = 0.1
Ttyp = 2000.0
#print(mdb)
lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray = initspec.init_premodit(
    mdb.nu_lines,
    nu_grid,
    mdb.elower,
    mdb.alpha_ref,
    mdb.n_Texp,
    mdb.Sij0,
    Ttyp,
    interval_contrast=interval_contrast,
    dit_grid_resolution=dit_grid_resolution,
    warning=False)


def frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini):
    g = 2478.57730044555 * Mp / Rp**2
    qtarr = vmap(mdb.qr_interp)(Tarr)
    xsm = xsmatrix(Tarr, Parr, R, pmarray, lbd, nu_grid, ngamma_ref_grid,
                   n_Texp_grid, multi_index_uniqgrid, elower_grid, molmassCH4, qtarr)
    dtaumCH4 = dtauM(dParr, jnp.abs(xsm), MMR_CH4 * np.ones_like(Parr), molmassCH4,
                     g)
    # CIA
    dtaucH2H2 = dtauCIA(nu_grid, Tarr, Parr, dParr, vmrH2, vmrH2, mmw, g,
                        cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)
    dtau = dtaumCH4 + dtaucH2H2
    sourcef = piBarr(Tarr, nu_grid)

    F0 = rtrun(dtau, sourcef) / norm
    Frot = response.rigidrot(nu_grid, F0, vsini, u1, u2)
    mu = response.ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV)
    return mu


# test
if False:
    Tarr = 1295.0 * (Parr / Pref)**0.1
    mu = frun(Tarr,
              MMR_CH4=0.0059,
              Mp=33.2,
              Rp=0.88,
              u1=0.0,
              u2=0.0,
              RV=10.0,
              vsini=20.0)
    plt.plot(wavd, mu)
    plt.show()
    np.savetxt("spectrum_ch4_new.txt",np.array([wavd,mu*norm]).T,delimiter=",")

    
Mp = 33.2


def model_c(nu1, y1):
    Rp = numpyro.sample('Rp', dist.Uniform(0.4, 1.2))
    RV = numpyro.sample('RV', dist.Uniform(5.0, 15.0))
    MMR_CH4 = numpyro.sample('MMR_CH4', dist.Uniform(0.0, 0.015))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0, 1500.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05, 0.2))
    vsini = numpyro.sample('vsini', dist.Uniform(15.0, 25.0))
    g = 2478.57730044555 * Mp / Rp**2  # gravity
    u1 = 0.0
    u2 = 0.0
    # T-P model//
    Tarr = T0 * (Parr / Pref)**alpha
    # line computation CH4
    mu = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)
    numpyro.sample('y1', dist.Normal(mu, sigmain), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000
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

pararr = ['Rp', 'T0', 'alpha', 'MMR_CH4', 'vsini', 'RV']
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True)
plt.show()
