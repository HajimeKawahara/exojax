""" Reverse modeling of CO/HITEMP using MODIT
    works with ExoJAX v1.6
"""

#!/usr/bin/env python
# coding: utf-8
import arviz
from exojax.opacity.modit.modit import setdgm_hitran
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from exojax.atm.atmprof import pressure_layer_logspace
from exojax.opacity import initspec
from exojax.rt.layeropacity import layer_optical_depth
from exojax.rt.layeropacity import layer_optical_depth_CIA
from exojax.opacity.modit import modit
from exojax.database.exomol.api import MdbExomol , contdb
from exojax.rt.rtransfer import rtrun_emis_pureabs_fbased2st
from exojax.rt import planck
from exojax.postproc.response import ipgauss_sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.database import molinfo 
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.constants import RJ
from exojax.utils.instfunc import resolution_to_gaussian_std

dat = pd.read_csv("spectrum_co.txt", delimiter=",", names=("wav", "flux"))
wavd = dat["wav"].values
flux = dat["flux"].values
nusd = jnp.array(1.0e8 / wavd[::-1])
sigmain = 0.05
norm = 20000
nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

NP = 100
Parr, dParr, k = pressure_layer_logspace(nlayer=NP)
Nx = 5000
nus, wav, res = wavenumber_grid(
    np.min(wavd) - 5.0, np.max(wavd) + 5.0, Nx, unit="AA", xsmode="modit"
)
Rinst = 100000.0
beta_inst = resolution_to_gaussian_std(Rinst)

molmassCO = molinfo.molmass_isotope("CO")
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass_isotope("H2")
vmrH2 = mmrH2 * mmw / molmassH2  # VMR

#
Mp = 33.2
mdbCO = MdbHitemp(".database/CO", nus, crit=1.0e-30, gpu_transfer=True)
cdbH2H2 = contdb.CdbCIA(".database/H2-H2_2011.cia", nus)
print("N=", len(mdbCO.nu_lines))

# Reference pressure for a T-P model
Pref = 1.0  # bar
ONEARR = np.ones_like(Parr)
ONEWAV = jnp.ones_like(nflux)

cnu, indexnu, R, pmarray = initspec.init_modit(mdbCO.nu_lines, nus)


# Precomputing gdm_ngammaL
def fT(T0, alpha):
    return T0[:, None] * (Parr[None, :] / Pref) ** alpha[:, None]


T0_test = np.array([1100.0, 1500.0, 1100.0, 1500.0])
alpha_test = np.array([0.2, 0.2, 0.05, 0.05])
res = 0.2
vmrCO_ref = 4.9e-4
dgm_ngammaL = setdgm_hitran(
    mdbCO, fT, Parr, Parr * vmrCO_ref, R, molmassCO, res, T0_test, alpha_test
)

# check dgm
if False:
    from exojax.plot.ditplot import plot_dgmn

    Tarr = 1300.0 * (Parr / Pref) ** 0.1
    SijM_CO, ngammaLM_CO, nsigmaDl_CO = modit.hitran(
        mdbCO, Tarr, Parr, Parr * vmrCO_ref, R, molmassCO
    )
    plot_dgmn(Parr, dgm_ngammaL, ngammaLM_CO, 0, 6)
    plt.show()

# settings before HMC
vsini_max = 100.0
vr_array = velocity_grid(res, vsini_max)


def frun(Tarr, MMR_CO, Mp, Rp, u1, u2, RV, vsini):
    g = 2478.57730044555 * Mp / Rp**2
    VMR_CO = MMR_CO * mmw / molmassCO
    SijM_CO, ngammaLM_CO, nsigmaDl_CO = modit.hitran(
        mdbCO, Tarr, Parr, Parr * VMR_CO, R, molmassCO
    )
    xsm_CO = modit.xsmatrix(
        cnu, indexnu, R, pmarray, nsigmaDl_CO, ngammaLM_CO, SijM_CO, nus, dgm_ngammaL
    )
    # abs is used to remove negative values in xsv
    dtaumCO = layer_optical_depth(dParr, jnp.abs(xsm_CO), MMR_CO * ONEARR, molmassCO, g)
    # CIA
    dtaucH2H2 = layer_optical_depth_CIA(
        nus,
        Tarr,
        Parr,
        dParr,
        vmrH2,
        vmrH2,
        mmw,
        g,
        cdbH2H2.nucia,
        cdbH2H2.tcia,
        cdbH2H2.logac,
    )
    dtau = dtaumCO + dtaucH2H2
    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun_emis_pureabs_fbased2st(dtau, sourcef) / norm
    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
    mu = ipgauss_sampling(nusd, nus, Frot, beta_inst, RV, vr_array)
    return mu


# test
if False:
    Tarr = 1200.0 * (Parr / Pref) ** 0.1
    mu = frun(
        Tarr, MMR_CO=0.0059, Mp=33.2, Rp=0.88, u1=0.0, u2=0.0, RV=10.0, vsini=20.0
    )
    plt.plot(wavd, mu)
    plt.show()

Mp = 33.2


def model_c(nu1, y1):
    Rp = numpyro.sample("Rp", dist.Uniform(0.4, 1.2))
    RV = numpyro.sample("RV", dist.Uniform(5.0, 15.0))
    MMR_CO = numpyro.sample("MMR_CO", dist.Uniform(0.0, 0.015))
    T0 = numpyro.sample("T0", dist.Uniform(1000.0, 1500.0))
    alpha = numpyro.sample("alpha", dist.Uniform(0.05, 0.2))
    vsini = numpyro.sample("vsini", dist.Uniform(15.0, 25.0))
    g = 2478.57730044555 * Mp / Rp**2  # gravity
    u1 = 0.0
    u2 = 0.0
    # T-P model//
    Tarr = T0 * (Parr / Pref) ** alpha
    # line computation CO
    mu = frun(Tarr, MMR_CO, Mp, Rp, u1, u2, RV, vsini)
    numpyro.sample("y1", dist.Normal(mu, sigmain), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 300, 600
# kernel = NUTS(model_c, forward_mode_differentiation=True)
kernel = NUTS(model_c, forward_mode_differentiation=False)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, nu1=nusd, y1=nflux)

# SAMPLING
posterior_sample = mcmc.get_samples()
pred = Predictive(model_c, posterior_sample, return_sites=["y1"])
predictions = pred(rng_key_, nu1=nusd, y1=None)
median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.9)

# PLOT
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6.0))
ax.plot(wavd[::-1], median_mu1, color="C0")
ax.plot(wavd[::-1], nflux, "+", color="black", label="data")
ax.fill_between(
    wavd[::-1],
    hpdi_mu1[0],
    hpdi_mu1[1],
    alpha=0.3,
    interpolate=True,
    color="C0",
    label="90% area",
)
plt.xlabel("wavelength ($\AA$)", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)

pararr = ["Rp", "T0", "alpha", "MMR_CO", "vsini", "RV"]
arviz.plot_pair(arviz.from_numpyro(mcmc), kind="kde", divergences=False, marginals=True)
plt.show()
