""" Reverse modeling of Methane emission spectrum using plg
"""

#!/usr/bin/env python
# coding: utf-8

SaveOrNot = False
path_fig = './'
Nth_run = 1
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import arviz
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
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
from exojax.spec import planck, response
from exojax.spec import molinfo
from exojax.spec.plg import MdbExomol_plg
from exojax.utils.constants import RJ
from exojax.utils.instfunc import R2STD
import numpy as np
from exojax.spec import initspec, modit
import pkg_resources
from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW
from exojax.spec.planck import piBarr
from jax import vmap

filename = pkg_resources.resource_filename(
    'exojax', 'data/testdata/' + SAMPLE_SPECTRA_CH4_NEW)
dat = pd.read_csv(filename, delimiter=",", names=("wav", "flux"))
wavd = dat['wav'].values[130:180] #trim
flux = dat['flux'].values[-180:-130] #trim
nusd = jnp.array(1.e8 / wavd[::-1])
sigmain = 0.03
norm = 20000
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

NP = 100
Parr, dParr, k = rtransfer.pressure_layer(NP=NP)
Nx = 250 #trim
nu_grid, wav, res = gen_wavenumber_grid(
                           np.min(wavd) - 2.0, #trim
                           np.max(wavd) + 2.0, #trim
                           Nx,
                           unit='AA',
                           xsmode='modit')
Rinst = 100000.
beta_inst = R2STD(Rinst)

molmassCH4 = molinfo.molmass('CH4')
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass('H2')
vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR

# Reference pressure for a T-P model
Pref = 1.0  # bar

# Rough guess on the typical atmospheric temperature
Tgue = 2000.0

# Load line database
mdb, cnu, indexnu = MdbExomol_plg('.database/CH4/12C-1H4/YT10to10/',
                                    nu_grid, Tgue, errTgue=200.0,
                                    threshold_persist_freezing=1000.,
                                    crit=1.e-30)
cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nu_grid)
print('N=', len(mdb.nu_lines))

#Precomputing dgm_ngammaL
cnu_dummy, indexnu_dummy, R, pmarray = initspec.init_modit(mdb.nu_lines, nu_grid)
fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
T0_test=np.array([1100., 1500., 1500., 1100.])
alpha_test=np.array([0.2,0.2,0.05,0.05])
res=0.2
dgm_ngammaL = modit.set_ditgrid_matrix_exomol(mdb, fT, Parr, R, molmassCH4, res, T0_test, alpha_test)


def frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini):
    g = 2478.57730044555 * Mp / Rp**2
    qtarr = vmap(mdb.qr_interp)(Tarr)
    SijM, ngammaLM, nsigmaDl = modit.exomol(mdb, Tarr, Parr, R, molmassCH4)
    xsm = modit.xsmatrix(cnu, indexnu, R, pmarray, nsigmaDl, ngammaLM, SijM, nu_grid, dgm_ngammaL)
    dtaumCH4 = dtauM(dParr, jnp.abs(xsm), MMR_CH4 * np.ones_like(Parr), molmassCH4, g)
    
    # CIA
    dtaucH2H2 = dtauCIA(nu_grid, Tarr, Parr, dParr, vmrH2, vmrH2, mmw, g,
                        cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)
    dtau = dtaumCH4 + dtaucH2H2
    sourcef = piBarr(Tarr, nu_grid)

    F0 = rtrun(dtau, sourcef) / norm
    Frot = response.rigidrot(nu_grid, F0, vsini, u1, u2)
    mu = response.ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV)
    return mu


# Make mock data
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
    plt.plot(wavd[::-1], mu)
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

#mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)#, num_chains=4, thinning=2)
mcmc.run(rng_key_, nu1=nusd, y1=nflux)
mcmc.print_summary()
if SaveOrNot:
    import pickle
    with open(path_fig + f'mcmc_{str(int(Nth_run))}.pkl', 'wb') as f:
        pickle.dump(mcmc, f)

# SAMPLING
posterior_sample = mcmc.get_samples()
pred = Predictive(model_c, posterior_sample, return_sites=['y1'])
predictions = pred(rng_key_, nu1=nusd, y1=None)
median_mu1 = jnp.median(predictions['y1'], axis=0)
hpdi_mu1 = hpdi(predictions['y1'], 0.9)

# PLOT
if SaveOrNot:
    arviz.plot_trace(mcmc)
    plt.savefig(path_fig + f"trace_{str(int(Nth_run))}.pdf")

    fig, ax = plt.subplots(figsize=(20, 3), tight_layout=True)
    ax.plot(np.arange(len(posterior_samples['T0'])),
            mcmc.get_extra_fields()['diverging'],
            ls='', marker='|', markersize=10.)
    ax.set_ylim(-1, 2)
    plt.savefig(path_fig + f"flag_divergence_{str(int(Nth_run))}.pdf")

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
if SaveOrNot:
    plt.savefig(path_fig + f"spectra_{str(int(Nth_run))}.pdf")

arviz.rcParams['plot.max_subplots'] = np.sum(np.arange(len(posterior_sample.keys())+1))
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True)
if SaveOrNot:
    plt.savefig(path_fig + f"pair_{str(int(Nth_run))}.pdf")
plt.show()

#SAVE
if SaveOrNot:
    np.savez(path_fig + f"savepos_{str(int(Nth_run))}.npz", [posterior_sample])
    np.savez(path_fig + f"saveplotpred_{str(int(Nth_run))}.npz", np.array([wavd, nflux, median_mu1, hpdi_mu1], dtype=object))
    import sys
    file_path = path_fig + f"summary_{str(int(Nth_run))}.txt"
    sys.stdout = open(file_path, "w")
    print(path_fig + '\n' + str(Nth_run))
    print("wls, wll:\t", min(wavd), '–', max(wavd))
    print("num_warmup, num_samples:\t" + str(int(num_warmup))+', '+str(int(num_samples)))
    mcmc.print_summary()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
