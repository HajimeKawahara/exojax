""" Reverse modeling of Methane transmission spectrum using PreMODIT
"""

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import pandas as pd
from importlib.resources import files

from exojax.rt import ArtTransPure
from exojax.database.exomol.api import MdbExomol
from exojax.opacity import OpaPremodit
from exojax.utils.grids import nu2wav
from exojax.utils.grids import wavenumber_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_TRANS
from exojax.utils.constants import RJ, Rs
from exojax.utils.astrofunc import gravity_jupiter
from exojax.postproc.specop import SopInstProfile


from jax import config

config.update("jax_enable_x64", True)

# PPL
import arviz
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

# loading the data
filename = files("exojax").joinpath("data/testdata/" + SAMPLE_SPECTRA_CH4_TRANS)
dat = pd.read_csv(filename, delimiter=",", names=("wavenumber", "flux"))
nusd = dat["wavenumber"].values
radius_ratio = dat["flux"].values
wavd = nu2wav(nusd, wavelength_order="ascending")


sigmain = 0.0001
radis_ratio_obs = radius_ratio  + np.random.normal(0, sigmain, len(wavd))

Nx = 7500
nu_grid, wav, res = wavenumber_grid(
    np.min(wavd) - 10.0,
    np.max(wavd) + 10.0,
    Nx,
    unit="AA",
    xsmode="premodit",
    wavelength_order="ascending",
)

T_fid = 500.0
Tlow = 400.0
Thigh = 700.0

art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0e1, nlayer=100)
art.change_temperature_range(Tlow, Thigh)

Rinst = 100000.0
beta_inst = resolution_to_gaussian_std(Rinst)

### CH4 setting (PREMODIT)
mdb = MdbExomol(".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False)

print("N=", len(mdb.nu_lines))
diffmode = 0
opa = OpaPremodit(
    mdb=mdb,
    nu_grid=nu_grid,
    diffmode=diffmode,
    auto_trange=[Tlow, Thigh],
    dit_grid_resolution=0.2,
    allow_32bit=True,
)

## CIA setting
# cdbH2H2 = CdbCIA(".database/H2-H2_2011.cia", nu_grid)
mu_fid = 2.2
# settings before HMC
radius_btm = RJ

vrmax = 100.0  # km/s
sop = SopInstProfile(nu_grid, vrmax)


def frun(T0, MMR_CH4, Mp, Rp, RV):
    gravity_btm = gravity_jupiter(Rp, Mp)

    # T-P model
    Tarr = T0 * jnp.ones_like(art.pressure)
    mmw_arr = mu_fid * np.ones_like(art.pressure)

    gravity = art.gravity_profile(Tarr, mmw_arr, radius_btm, gravity_btm)
    mmr_arr = art.constant_profile(MMR_CH4)

    # molecule
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

    Rp2 = art.run(dtau, Tarr, mmw_arr, radius_btm, gravity_btm)
    mu = sop.sampling(Rp2, RV, nusd)

    return jnp.sqrt(mu) * radius_btm / Rs


import matplotlib.pyplot as plt

# g = gravity_jupiter(0.88, 33.2)
Rp = 1.0
Mp = 1.0
MMR_CH4 = 0.0059
RV = 10.0
T0 = 500.0
radius_ratio_test = frun(T0, MMR_CH4, Mp, Rp, RV)

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(nusd, radis_ratio_obs)
ax = fig.add_subplot(212)
plt.plot(nusd, radius_ratio_test, ls="dashed")
plt.savefig("spectrum_reverse.png")

def model_c(y1):
    RV = numpyro.sample("RV", dist.Uniform(5.0, 15.0))
    Rp = numpyro.sample("Rp", dist.Uniform(0.8, 1.2))
    Mp = numpyro.sample("Mp", dist.Uniform(0.8, 1.2))
    MMR_CH4 = numpyro.sample("MMR_CH4", dist.Uniform(0.0, 0.01))
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    mu = frun(T0, MMR_CH4, Mp, Rp, RV)
    numpyro.sample("y1", dist.Normal(mu, sigmain), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 500, 1000
# kernel = NUTS(model_c, forward_mode_differentiation=True)
kernel = NUTS(model_c, forward_mode_differentiation=False)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, y1=radis_ratio_obs)
mcmc.print_summary()

# SAMPLING
posterior_sample = mcmc.get_samples()
pred = Predictive(model_c, posterior_sample, return_sites=["y1"])
predictions = pred(rng_key_, y1=None)
median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.9)

# PLOT
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6.0))
ax.plot(wavd[::-1], median_mu1, color="C0")
ax.plot(wavd[::-1], radis_ratio_obs, "+", color="black", label="data")
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
plt.savefig("pred_diffmode" + str(diffmode) + ".png")
plt.close()

pararr = ["Rp", "T0", "alpha", "MMR_CH4", "vsini", "RV"]
arviz.plot_pair(arviz.from_numpyro(mcmc), kind="kde", divergences=False, marginals=True)
plt.savefig("corner_diffmode" + str(diffmode) + ".png")
# plt.show()
