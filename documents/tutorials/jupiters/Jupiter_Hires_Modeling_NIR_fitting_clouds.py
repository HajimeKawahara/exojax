#!/usr/bin/env python
# coding: utf-8

# # Modeling a High Resolution Reflection Spectrum

# Hajime Kawahara July 13th (2024)
# This code analyzes the reflection spectrum of Jupiter. We here try to solve two problems.
# One is we need to calibrate the wavenumber grid of the data because the calibration lines were insufficient and the wavelength of the data is not accurate.
# To do so, we use the reflection spectrum model itself.
# The other is, after the wavelenghtcalibration, we try to fit the model to the calibrated data.
#
# This note coresponds to the other one, using the output of the code for the former one (Jupiter_Hires_Modeling_NIR_fitting.ipynb)

# %%

### Preparation
# if this is the first run, set miegird_generate = True, and run the code to generate Mie grid. After that, set False.
miegird_generate = False
# when the optimization is performed, set opt_perform = True, after performing it, set False
opt_perform = True
# when HMC is performed, set hmc_perform = True, after performing it, set False
hmc_perform = False
###

import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import plotjupiter
import miegrid_generate

import pickle
import numpy as np
import matplotlib.pyplot as plt
from exojax.spec.unitconvert import nu2wav
import pandas as pd

from exojax.utils.constants import c  # light speed in km/s
from exojax.atm.idealgas import number_density
from exojax.utils.constants import m_u
from exojax.spec.unitconvert import wav2nu
from exojax.utils.grids import wavenumber_grid
from exojax.spec.atmrt import ArtReflectPure
from exojax.utils.astrofunc import gravity_jupiter
from exojax.spec.pardb import PdbCloud
from exojax.atm.atmphys import AmpAmcloud
from exojax.utils.zsol import nsol
from exojax.atm.mixratio import vmr2mmr
from exojax.spec.molinfo import molmass_isotope
from exojax.spec.opacont import OpaMie
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec.specop import SopInstProfile
from exojax.spec.opacalc import OpaPremodit
from jax import random
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from jaxopt import OptaxSolver
import optax
from jovispec.tpio import read_tpprofile_jupiter

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)
figshow = True  # False # if True, the figures are shown


# Forget about the following code. I need this to run the code somewhere...
# username="exoplanet01"
username = "kawahara"
if username == "exoplanet01":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


dat = np.loadtxt("jupiter_corrected.dat")  # made by Jupiter_Hires_Modeling_NIR.ipynb
unmask_nus_obs = dat[:, 0]
unmask_spectra = dat[:, 1]
mask = (unmask_nus_obs < 6163.5) + ((unmask_nus_obs > 6166) & (unmask_nus_obs < 6184.5))
nus_obs = unmask_nus_obs[mask]
wav_obs = nu2wav(nus_obs, unit="AA")

plotjupiter.print_wavminmax(wav_obs)

spectra = unmask_spectra[mask]

if figshow:
    plotjupiter.plot_spec1(unmask_nus_obs, unmask_spectra, nus_obs, spectra)

# ## Solar spectrum
# This is the reflected "solar" spectrum by Jupiter! So, we need the solar spectrum template.
#
# I found very good one: High-resolution solar spectrum taken from Meftar et al. (2023)
#
# - 10.21413/SOLAR-HRS-DATASET.V1.1_LATMOS
# - http://doi.latmos.ipsl.fr/DOI_SOLAR_HRS.v1.1.html
# - http://bdap.ipsl.fr/voscat_en/solarspectra.html
#

# filename = "/home/kawahara/solar-hrs/Spectre_HR_LATMOS_Meftah_V1.txt"
filename = "/home/" + username + "/solar-hrs/Spectre_HR_LATMOS_Meftah_V1.txt"
dat = pd.read_csv(filename, names=("wav", "flux"), comment=";", delimiter="\t")
dat["wav"] = dat["wav"] * 10

wav_solar = dat["wav"][::-1]
solspec = dat["flux"][::-1]
nus_solar = wav2nu(wav_solar, unit="AA")

vrv = 10.0
vperc = vrv / 300000


if figshow:
    plotjupiter.plot_spec2(nus_obs, spectra, solspec, nus_solar, vperc)

nus, wav, res = wavenumber_grid(
    np.min(nus_obs) - 5.0, np.max(nus_obs) + 5.0, 10000, xsmode="premodit", unit="cm-1"
)

# read the temperature-pressure profile of Jupiter
dat = read_tpprofile_jupiter()

torig = dat["Temperature (K)"]
porig = dat["Pressure (bar)"]

# %%
art = ArtReflectPure(nu_grid=nus, pressure_btm=3.0e1, pressure_top=1.0e-3, nlayer=100)

# custom temperature profile
Parr = art.pressure
Tarr_np = np.interp(Parr, porig, torig)
i = np.argmin(Tarr_np)
Tarr_np[0:i] = Tarr_np[i]

# numpy ndarray should be converted to jnp.array (equivalent to Tarr=jnp.array(Tarr))
Tarr = art.custom_temperature(Tarr_np)

if True:
    plotjupiter.plottp(torig, porig, Parr, Tarr)

# %%
mu = 2.22  # mean molecular weight NASA Jupiter fact sheet
gravity = gravity_jupiter(1.0, 1.0)

if True:
    plotjupiter.plotPT(art, Tarr)

pdb_nh3 = PdbCloud("NH3")
amp_nh3 = AmpAmcloud(pdb_nh3, bkgatm="H2")
amp_nh3.check_temperature_range(Tarr)

# condensate substance density
rhoc = pdb_nh3.condensate_substance_density  # g/cc
n = nsol()
abundance_nh3 = 3.0 * n["N"]  # x 3 solar abundance
molmass_nh3 = molmass_isotope("NH3", db_HIT=False)


# fsed = 10.
fsed = 3.0
sigmag = 2.0
Kzz = 1.0e4
MMRbase_nh3 = vmr2mmr(abundance_nh3, molmass_nh3, mu)

rg_layer, MMRc = amp_nh3.calc_ammodel(
    Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
)


# search for rg, sigmag range based on fsed and Kzz range
fsed_range = [0.1, 10.0]
Kzz_range = [1.0e2, 1.0e6]


if miegird_generate:
    Kzz, rg_layer, MMRc = miegrid_generate.generate_miegrid_new(
        Tarr,
        Parr,
        mu,
        gravity,
        pdb_nh3,
        amp_nh3,
        molmass_nh3,
        sigmag,
        MMRbase_nh3,
        fsed_range,
        Kzz_range,
    )
else:
    pdb_nh3.load_miegrid()


# to convert MMR to g/L ...
fac = molmass_nh3 * m_u * number_density(Parr, Tarr) * 1.0e3  # g/L

if figshow:
    plotjupiter.plot_cloud_structure(Parr, rg_layer, MMRc, fac)

fsed = 3.0
rg_layer, MMRc = amp_nh3.calc_ammodel(
    Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
)
rg = np.nanmean(rg_layer)

opa_nh3 = OpaMie(pdb_nh3, nus)

sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector(
    rg, sigmag
)  # if using MieGrid
# sigma_extinction, sigma_scattering, asymmetric_factor = (
#    opa_nh3.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
# ) # direct computation

# plt.plot(pdb_nh3.refraction_index_wavenumber, miepar[50,:,0])

if figshow:
    plotjupiter.plot_extinction(
        nus, sigma_extinction, sigma_scattering, asymmetric_factor
    )

# Next, we consider the gas component, i.e. methane
from exojax.spec.api import MdbHitemp

Eopt = 3300.0  # this is from the Elower optimization result
# mdb_reduced = MdbHitemp("CH4", nurange=[nus_start,nus_end], isotope=1, elower_max=Eopt)
mdb_reduced = MdbHitemp("CH4", nurange=[nus[0], nus[-1]], isotope=1, elower_max=Eopt)


opa = OpaPremodit(
    mdb_reduced, nu_grid=nus, allow_32bit=True, auto_trange=[80.0, 300.0]
)  # this reduced the device memory use in 5/6.

## Spectrum Model
nusjax = jnp.array(nus)
nusjax_solar = jnp.array(nus_solar)
solspecjax = jnp.array(solspec)


# ### gas opacity
molmass_ch4 = molmass_isotope("CH4", db_HIT=False)
asymmetric_parameter = asymmetric_factor + np.zeros((len(art.pressure), len(nus)))
reflectivity_surface = np.zeros(len(nus))

sop = SopInstProfile(nus)
# log_fsed, log_Kzz, vrv, vv, boradening, mmr, normalization factor
#parinit = jnp.array(
#    [np.log10(1.0) * 10000.0, np.log10(1.0e4) * 10.0, -5.0, -55.0, 2.5, 2.0, 0.6]
#)
sigmag = 2.0
parinit = jnp.array(
    [np.log10(10.0) * 1e4, np.log10(1.0e3) * 10.0, -5.0, -55.0, 2.5, 2.0, 2.5]
)
multiple_factor = jnp.array([1e-4, 0.1, 1.0, 1.0, 10000.0, 0.01, 1.0])


def spectral_model(params):

    log_fsed, log_Kzz, vrv, vv, _broadening, const_mmr_ch4, factor = (
        params * multiple_factor
    )

    fsed = 10**log_fsed
    Kzz = 10**log_Kzz
    broadening = 25000.0

    # temperatures
    # cloud
    rg_layer, MMRc = amp_nh3.calc_ammodel(
        Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
    )
    rg = jnp.mean(rg_layer)

    ### this one
    sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
    #sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector(
    #    rg, sigmag
    #)
    dtau_cld = art.opacity_profile_cloud_lognormal(
        sigma_extinction, rhoc, MMRc, rg, sigmag, gravity
    )
    dtau_cld_scat = art.opacity_profile_cloud_lognormal(
        sigma_scattering, rhoc, MMRc, rg, sigmag, gravity
    )

    asymmetric_parameter = asymmetric_factor + np.zeros((len(art.pressure), len(nus)))

    # opacity
    mmr_ch4 = art.constant_mmr_profile(const_mmr_ch4)
    xsmatrix = opa.xsmatrix(Tarr, Parr)
    dtau_ch4 = art.opacity_profile_xs(xsmatrix, mmr_ch4, molmass_ch4, gravity)

    single_scattering_albedo = (dtau_cld_scat) / (dtau_cld + dtau_ch4)
    dtau = dtau_cld + dtau_ch4

    # velocity
    vpercp = (vrv + vv) / c
    incoming_flux = jnp.interp(nusjax, nusjax_solar * (1.0 + vpercp), solspecjax)

    Fr = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        reflectivity_surface,
        incoming_flux,
    )

    std = resolution_to_gaussian_std(broadening)
    Fr_inst = sop.ipgauss(Fr, std)
    Fr_samp = sop.sampling(Fr_inst, vv, nus_obs)
    return factor * Fr_samp


def unpack_params(params):
    return params * multiple_factor



print(parinit)    
F_samp_init = spectral_model(unpack_params(parinit))
print("(*_*)")

if figshow:
    plt = plotjupiter.plot_opt(nus_obs, spectra, F_samp_init, F_samp_init)
    plt.show()

import sys
sys.exit()


def cost_function(params):
    return jnp.sum((spectra - spectral_model(params)) ** 2)


if opt_perform:
    adam = OptaxSolver(opt=optax.adam(1.0e-2), fun=cost_function)
    res = adam.run(parinit)

    # res.params
    print(unpack_params(res.params))
    print(unpack_params(parinit))

    F_samp = spectral_model(res.params)
    F_samp_init = spectral_model(parinit)

    plotjupiter.plot_opt(nus_obs, spectra, F_samp_init, F_samp)

    print(res.params)
    plt.show()
    import sys

    sys.exit()


# %%

# T0, log_fsed, log_Kzz, vrv, vv, boradening, mmr, normalization factor
# parinit = jnp.array(
#    [1.5, np.log10(1.0) * 10000.0, np.log10(1.0e4) * 10.0, -5.0, -55.0, 2.5, 1.2, 0.6]
# )
# multiple_factor = jnp.array([100.0, 0.0001, 0.1, 1.0, 1.0, 10000.0, 0.01, 1.0])


def model_c(nu1, y1):

    # T0_n = numpyro.sample("T0_n", dist.Uniform(0.5, 2.0))
    log_fsed_n = numpyro.sample("log_fsed_n", dist.Uniform(-1.0e4, 1.0e4))
    log_Kzz_n = numpyro.sample("log_Kzz_n", dist.Uniform(30.0, 50.0))
    vrv = numpyro.sample("vrv", dist.Uniform(-10.0, 10.0))
    vv = numpyro.sample("vv", dist.Uniform(-70.0, -40.0))
    broadening = 25000.0  # fix
    molmass_ch4_n = numpyro.sample("MMR_CH4_n", dist.Uniform(0.0, 5.0))
    factor = numpyro.sample("factor", dist.Uniform(0.0, 1.0))

    params = jnp.array(
        [T0_n, log_fsed_n, log_Kzz_n, vrv, vv, broadening, molmass_ch4_n, factor]
    )
    mean = spectral_model(params)

    sigma = numpyro.sample("sigma", dist.Exponential(10.0))
    err_all = jnp.ones_like(nu1) * sigma
    # err_all = jnp.sqrt(y1err**2. + sig**2.)
    numpyro.sample("y1", dist.Normal(mean, err_all), obs=y1)


# initialization
import jax.numpy as jnp

if opt_perform:
    # T0, log_fsed, log_Kzz, vrv, vv, boradening (fix), mmr, normalization factor
    #    init_params = {
    #        "T0_n": 1.5,
    #        }
    init_params = {
        "T0_n": 0.89,
        "log_fsed_n": 0.44,
        "log_Kzz_n": 40.0,
        "vrv": -1.1,
        "vv": -58.3,
        "MMR_CH4_n": 1.58,
        "factor": 0.597,
        "sigma": 1.0,
    }

    #    init_params = {
#        "T0_n": jnp.array(res.params[0]),
#        "log_fsed_n": jnp.array(res.params[1]),
#        "log_Kzz_n": jnp.array(res.params[2]),
#        "vrv": jnp.array(res.params[3]),
#        "vv": jnp.array(res.params[4]),
#        "MMR_CH4_n":jnp.array(res.params[6]),
#        "factor":jnp.array(res.params[7])
#        }
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

if hmc_perform:
    print("HMC starts")
    # num_warmup, num_samples = 500, 1000
    ######
    num_warmup, num_samples = 100, 200
    ######
    # kernel = NUTS(model_c,forward_mode_differentiation=True)
    kernel = NUTS(model_c)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    if opt_perform == True:
        mcmc.run(rng_key_, nu1=nus_obs, y1=spectra, init_params=init_params)
    #        mcmc.run(rng_key_, nu1=nus_obs, y1=spectra)
    else:
        mcmc.run(rng_key_, nu1=nus_obs, y1=spectra)
    mcmc.print_summary()

    with open("output/samples.pickle", mode="wb") as f:
        pickle.dump(mcmc.get_samples(), f)

with open("output/samples.pickle", mode="rb") as f:
    samples = pickle.load(f)

print("prediction starts")
pred = Predictive(model_c, samples, return_sites=["y1"])
predictions = pred(rng_key_, nu1=nus_obs, y1=None)
median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.95)

# prediction plot
plotjupiter.plot_prediction(wav_obs, spectra, median_mu1, hpdi_mu1)


if hmc_perform:
    np.savez("output/all.npz", [median_mu1, hpdi_mu1])

#####################################################
