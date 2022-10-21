#!/usr/bin/env python
# coding: utf-8
import arviz
from jax import jit, vmap
from exojax.spec.modit import setdgm_exomol
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
from jax import random
import pandas as pd
import numpy as np
from jax import jit
import matplotlib.pyplot as plt
import jax.numpy as jnp
from exojax.spec import rtransfer as rt
from exojax.spec import dit, modit
from exojax.spec import moldb, contdb
from exojax.spec.exomol import gamma_exomol
from exojax.spec import gamma_natural
from exojax.spec.hitran import SijT
from exojax.spec.dit import npgetix
from exojax.spec import rtransfer as rt
from exojax.utils.grids import wavenumber_grid
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, wavenumber_grid
from exojax.spec import planck, response
from exojax.spec.lpf import xsvector
from exojax.spec import molinfo
from exojax.utils.constants import RJ, pc, Rs, c
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec import normalized_doppler_sigma
import numpy as np
from exojax.spec import initspec
from scipy.signal import medfilt
from scipy.stats import median_absolute_deviation as mad

norm = 20000.
dat = pd.read_csv('datJ1645/order9.txt', delimiter=',', names=('wav', 'flux'))
wavd = dat['wav'].values*1.e1  # AA
flux = dat['flux'].values
plt.plot(wavd, flux, alpha=0.4, label='raw')
# edge removal
ts = 10
te = -100
wavd = wavd[ts:te]
flux = flux[ts:te]

# outlier
md = medfilt(flux, kernel_size=17)  # IRD/MMF
# md=medfilt(flux,kernel_size=7) #REACH

medf = flux-md
plt.plot(wavd, medf, color='gray', alpha=0.4, label='flux-median_filt')
sn = 5.0

plt.axhline(sn*mad(medf), color='gray', ls='dashed', alpha=0.4)
mask = np.abs(medf-np.median(medf)) < sn*mad(medf)
plt.plot(wavd[mask], medf[mask], '+', color='C5',
         alpha=0.4, label='flux-median_filt')

# Wavelength mask
mask = mask*(wavd < 15690.)
###

flux = flux[mask]
wavd = wavd[mask]
plt.plot(wavd, flux, alpha=0.7, color='C2', label='cleaned')
plt.legend()
plt.ylim(-1000, 5000)
plt.show()

nflux = flux[::-1]/np.median(flux)
nusd = jnp.array(1.e8/wavd[::-1])


NP = 100
Parr, dParr, k = rt.pressure_layer(NP=NP)
Nx = 5000
nus, wav, res = wavenumber_grid(np.min(wavd)-10.0, np.max(wavd) +
                       10.0, Nx, unit='AA', xsmode='modit')
Rinst = 100000.
beta_inst = resolution_to_gaussian_std(Rinst)


molmassH2O = molinfo.molmass('H2O')
molmassCO = molinfo.molmass('CO')

mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass('H2')
vmrH2 = (mmrH2*mmw/molmassH2)  # VMR

#
Mp = 33.2
mdbH2O = moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL/', nus, crit=1.e-50)
mdbCO = moldb.MdbExomol('.database/CO/12C-16O/Li2015/', nus)
cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nus)
print('N=', len(mdbH2O.nu_lines))

# Reference pressure for a T-P model
Pref = 1.0  # bar
ONEARR = np.ones_like(Parr)
ONEWAV = jnp.ones_like(nflux)

#

cnu_H2O, indexnu_H2O, R_H2O, pmarray_H2O = initspec.init_modit(
    mdbH2O.nu_lines, nus)
cnu_CO, indexnu_CO, R_CO, pmarray_CO = initspec.init_modit(mdbCO.nu_lines, nus)
R = R_CO
# Precomputing gdm_ngammaL


def fT(T0, alpha): return T0[:, None]*(Parr[None, :]/Pref)**alpha[:, None]


T0_test = np.array([1100.0, 1500.0, 1100.0, 1500.0])
alpha_test = np.array([0.2, 0.2, 0.05, 0.05])
res = 0.2
dgm_ngammaL_H2O = setdgm_exomol(
    mdbH2O, fT, Parr, R_H2O, molmassH2O, res, T0_test, alpha_test)
dgm_ngammaL_CO = setdgm_exomol(
    mdbCO, fT, Parr, R_CO, molmassCO, res, T0_test, alpha_test)

# check dgm
if False:
    from exojax.plot.ditplot import plot_dgmn
    Tarr = 1300.*(Parr/Pref)**0.1
    SijM_H2O, ngammaLM_H2O, nsigmaDl_H2O = modit.exomol(
        mdbH2O, Tarr, Parr, R_H2O, molmassH2O)
    SijM_CO, ngammaLM_CO, nsigmaDl_CO = modit.exomol(
        mdbCO, Tarr, Parr, R_CO, molmassCO)
    plot_dgmn(Parr, dgm_ngammaL, ngammaLM_H2O, 0, 6)
    plt.show()

# a core driver


def frun(Tarr, MMR_H2O, MMR_CO, Mp, Rp, u1, u2, RV, vsini):
    g = 2478.57730044555*Mp/Rp**2
    SijM_H2O, ngammaLM_H2O, nsigmaDl_H2O = modit.exomol(
        mdbH2O, Tarr, Parr, R, molmassH2O)
    xsm_H2O = modit.xsmatrix(cnu_H2O, indexnu_H2O, R_H2O, pmarray_H2O,
                             nsigmaDl_H2O, ngammaLM_H2O, SijM_H2O, nus, dgm_ngammaL_H2O)
    dtaumH2O = dtauM(dParr, jnp.abs(xsm_H2O), MMR_H2O*ONEARR, molmassH2O, g)

    SijM_CO, ngammaLM_CO, nsigmaDl_CO = modit.exomol(
        mdbCO, Tarr, Parr, R, molmassCO)
    xsm_CO = modit.xsmatrix(cnu_CO, indexnu_CO, R_CO, pmarray_CO,
                            nsigmaDl_CO, ngammaLM_CO, SijM_CO, nus, dgm_ngammaL_CO)
    dtaumCO = dtauM(dParr, jnp.abs(xsm_CO), MMR_CO*ONEARR, molmassCO, g)

    # CIA
    dtaucH2H2 = dtauCIA(nus, Tarr, Parr, dParr, vmrH2, vmrH2,
                        mmw, g, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)
    dtau = dtaumH2O+dtaumCO+dtaucH2H2
    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)/norm
    Frot = response.rigidrot(nus, F0, vsini, u1, u2)
    mu = response.ipgauss_sampling(nusd, nus, Frot, beta_inst, RV)
    return mu


# test
if True:
    MMR_H2O = 0.005  # mass mixing ratio
    MMR_CO = 0.01  # mass mixing ratio
    T0 = 1695.0  # K
    Tarr = T0*(Parr/Pref)**0.1
    mu = frun(Tarr, MMR_H2O=MMR_H2O, MMR_CO=MMR_CO, Mp=33.2,
              Rp=0.88, u1=0.0, u2=0.0, RV=50.0, vsini=10.0)
    plt.plot(wavd[::-1], mu/np.median(mu))
    plt.plot(wavd[::-1], nflux, alpha=0.3)
    plt.show()

nn = np.median(mu)
Mp = 33.2  # companion mass (assumption)


def model_c(nu1, y1):
    Rp = numpyro.sample('Rp', dist.Uniform(0.4, 1.2))
    RV = numpyro.sample('RV', dist.Uniform(40.0, 60.0))
    MMR_H2O = numpyro.sample('MMR_H2O', dist.Uniform(0.0, 0.015))
    MMR_CO = numpyro.sample('MMR_CO', dist.Uniform(0.0, 0.03))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0, 1700.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05, 0.2))
    vsini = numpyro.sample('vsini', dist.Uniform(1.0, 20.0))
    sigma = numpyro.sample('sigma', dist.Exponential(100.0))
    g = 2478.57730044555*Mp/Rp**2  # gravity
    u1 = 0.0
    u2 = 0.0
    # T-P model//
    Tarr = T0*(Parr/Pref)**alpha
    # line computation H2O
    mu = frun(Tarr, MMR_H2O, MMR_CO, Mp, Rp, u1, u2, RV, vsini)
    mu = mu/nn
    numpyro.sample('y1', dist.Normal(mu, sigmain), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 300, 600
kernel = NUTS(model_c, forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, nu1=nusd, y1=nflux)

# SAMPLING
posterior_sample = mcmc.get_samples()
np.savez('npz/savepos.npz', [posterior_sample])


pred = Predictive(model_c, posterior_sample, return_sites=['y1'])
predictions = pred(rng_key_, nu1=nusd, y1=None)
median_mu1 = jnp.median(predictions['y1'], axis=0)
hpdi_mu1 = hpdi(predictions['y1'], 0.9)

err = np.ones_like(nflux)
np.savez('npz/saveplotpred.npz', [wavd, nflux, err, median_mu1, hpdi_mu1])


# PLOT
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6.0))
ax.plot(wavd[::-1], median_mu1, color='C0')
ax.plot(wavd[::-1], nflux, '+', color='black', label='data')
ax.fill_between(wavd[::-1], hpdi_mu1[0], hpdi_mu1[1],
                alpha=0.3, interpolate=True, color='C0', label='90% area')
plt.xlabel('wavelength ($\AA$)', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig('pred.png')
# plt.show()

rc = {
    'plot.max_subplots': 250,
}

arviz.plot_pair(arviz.from_numpyro(mcmc), kind='kde',
                divergences=False, marginals=True)
plt.savefig('corner.png')
plt.show()
