#!/usr/bin/env python
# coding: utf-8

## Modeling a High Resolution Reflection Spectrum

# Hajime Kawahara September 2nd (2024)
# This code analyzes the reflection spectrum of Jupiter. We here try to solve two problems.
# One is we need to calibrate the wavenumber grid of the data because the calibration lines were insufficient and the wavelength of the data is not accurate.
# To do so, we use the reflection spectrum model itself.
# The other is, after the wavelenghtcalibration, we try to fit the model to the calibrated data.
# This note coresponds to the other one, using the output of the code for the former one (Jupiter_Hires_Modeling_NIR_fitting.ipynb)

### Preparation
# RT model
rtmode = "reflect"  # uses ArtReflectPure
# rtmode = "abs"  # uses ArtAbsPure

# if this is the first run, set miegird_generate = True, and run the code to generate Mie grid. After that, set False.
miegird_generate = False
# when the optimization is performed, set opt_perform = True, after performing it, set False
opt_perform = False
# checking atmosphere states
check_atm = False
# when HMC is performed, set hmc_perform = True, after performing it, set False
hmc_perform = True
use_init = False  # uses the initial values (obtained from optimization)
# if True, the figures are shown
figshow = False
###
import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import plotjupiter
from loaddata import load_jupiter_reflection
import miegrid_generate
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from exojax.spec.unitconvert import nu2wav
from exojax.spec.unitconvert import wav2nu
from exojax.utils.constants import c  # light speed in km/s
from exojax.utils.grids import wavenumber_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.spec.atmrt import ArtReflectPure
from exojax.spec.atmrt import ArtAbsPure
from exojax.spec.pardb import PdbCloud
from exojax.atm.atmphys import AmpAmcloud
from exojax.utils.zsol import nsol
from exojax.atm.atmconvert import vmr_to_mmr
from exojax.spec.molinfo import molmass_isotope
from exojax.spec.opacont import OpaMie
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec.specop import SopInstProfile
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.api import MdbHitemp
from exojax.spec.api import MdbExomol
from exojax.utils.constants import RJ

import arviz
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from jaxopt import OptaxSolver
import optax
from jovispec.tpio import read_tpprofile_jupiter
import tqdm

from jax import random
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


# Forget about the following code. I need this to run the code somewhere...
# username="exoplanet01"
username = "kawahara"
if username == "exoplanet01":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

wav_obs, nus_obs, spectra, unmask_wav_obs, unmask_nus_obs, unmask_spectra, mask = (
    load_jupiter_reflection()
)
plotjupiter.print_wavminmax(wav_obs)

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

# %% choose RT model
if rtmode == "reflect":
    art = ArtReflectPure(
        nu_grid=nus, pressure_btm=3.0e1, pressure_top=1.0e-3, nlayer=200
    )
elif rtmode == "abs":
    art = ArtAbsPure(nu_grid=nus, pressure_btm=3.0e1, pressure_top=1.0e-3, nlayer=200)
else:
    raise ValueError("rtmode is not correct")

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
n = nsol("AG89")
abundance_nh3 = 3.0 * n["N"]  # x 3 solar abundance
molmass_nh3 = molmass_isotope("NH3", db_HIT=False)
MMRbase_nh3 = vmr_to_mmr(abundance_nh3, molmass_nh3, mu)

# search for rg, sigmag range based on fsed and Kzz range
fsed_range = [0.1, 10.0]
Kzz_fixed = 1.0e4
sigmag_fixed = 2.0
vrv_fixed = 0.0

if miegird_generate:
    Kzz, rg_layer, MMRc = miegrid_generate.generate_miegrid_new(
        Tarr,
        Parr,
        mu,
        gravity,
        pdb_nh3,
        amp_nh3,
        molmass_nh3,
        sigmag_fixed,
        MMRbase_nh3,
        fsed_range,
        Kzz_fixed,
    )
    import sys

    sys.exit()
else:
    pdb_nh3.load_miegrid()

opa_nh3 = OpaMie(pdb_nh3, nus)

# Next, we consider the gas component, i.e. methane
Eopt = 3300.0  # this is from the Elower optimization result

# HITEMP or ExoMol/MM
# mdb_reduced = MdbHitemp("CH4", nurange=[nus[0], nus[-1]], isotope=1, elower_max=Eopt)
mdb_reduced = MdbExomol("CH4/12C-1H4/MM/", nurange=[nus[0], nus[-1]], elower_max=Eopt)

opa = OpaPremodit(
    mdb_reduced, nu_grid=nus, allow_32bit=True, auto_trange=[80.0, 300.0]
)  # this reduced the device memory use in 5/6.

## Spectrum Model
nusjax = jnp.array(nus)
nusjax_solar = jnp.array(nus_solar)
solspecjax = jnp.array(solspec)


# ### gas opacity
molmass_ch4 = molmass_isotope("CH4", db_HIT=False)
# asymmetric_parameter = asymmetric_factor + np.zeros((len(art.pressure), len(nus)))
reflectivity_surface = np.zeros(len(nus))

sop = SopInstProfile(nus)

broadening = 25000.0

if rtmode == "reflect":
    from model_reflect import unpack_params

    # log_fsed, sigmag, log_Kzz, vrv, vv, boradening, mmr, normalization factor
    parinit = jnp.array(
        [jnp.log10(3.0), sigmag_fixed, jnp.log10(Kzz_fixed), -5.0, -55.0, 2.5, 1.0, 0.6]
    )

    def atmospheric_model(params):
        # unused parameters are marked with _
        fsed, _sigmag, _Kzz, _vrv, vv, _broadening, const_mmr_ch4, factor = (
            unpack_params(params)
        )

        broadening = 25000.0
        rg_layer, MMRc = amp_nh3.calc_ammodel(
            Parr,
            Tarr,
            mu,
            molmass_nh3,
            gravity,
            fsed,
            sigmag_fixed,
            Kzz_fixed,
            MMRbase_nh3,
        )
        rg = jnp.mean(rg_layer)

        ### this one
        # sigma_extinction, sigma_scattering, asymmetric_factor = (
        #    opa_nh3.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
        # )
        sigma_extinction, sigma_scattering, asymmetric_factor = (
            opa_nh3.mieparams_vector(rg, sigmag_fixed)
        )
        dtau_cld = art.opacity_profile_cloud_lognormal(
            sigma_extinction, rhoc, MMRc, rg, sigmag_fixed, gravity
        )
        dtau_cld_scat = art.opacity_profile_cloud_lognormal(
            sigma_scattering, rhoc, MMRc, rg, sigmag_fixed, gravity
        )

        asymmetric_parameter = asymmetric_factor + np.zeros(
            (len(art.pressure), len(nus))
        )

        dtau_ch4 = methane_opacity(const_mmr_ch4)
        single_scattering_albedo = (dtau_cld_scat) / (dtau_cld + dtau_ch4)
        dtau = dtau_cld + dtau_ch4
        return (
            vv,
            factor,
            broadening,
            asymmetric_parameter,
            single_scattering_albedo,
            dtau,
        )

elif rtmode == "abs":
    from model_abs import unpack_params

    # log_surface pressure, vrv, vv, boradening, mmr, normalization factor
    parinit = jnp.array([np.log10(1.0), -5.0, -55.0, 2.5, 0.2, 0.6])

    def atmospheric_model(params):
        # unused parameters are marked with _
        _surface_pressure, _vrv, vv, _broadening, const_mmr_ch4, factor = unpack_params(
            params
        )
        surface_pressure = 0.3  # fix
        dtau_ch4 = methane_opacity(const_mmr_ch4)
        return vv, factor, broadening, surface_pressure, dtau_ch4


def methane_opacity(const_mmr_ch4):
    mmr_ch4 = art.constant_mmr_profile(const_mmr_ch4)
    xsmatrix = opa.xsmatrix(Tarr, Parr)
    dtau_ch4 = art.opacity_profile_xs(xsmatrix, mmr_ch4, molmass_ch4, gravity)
    return dtau_ch4


def spectral_model(params):
    if rtmode == "reflect":
        vv, factor, broadening, asymmetric_parameter, single_scattering_albedo, dtau = (
            atmospheric_model(params)
        )
    elif rtmode == "abs":
        vv, factor, broadening, surface_pressure, dtau = atmospheric_model(params)

    # velocity
    vpercp = (vrv_fixed + vv) / c
    incoming_flux = jnp.interp(nusjax, nusjax_solar * (1.0 + vpercp), solspecjax)

    if rtmode == "reflect":
        Fr = art.run(
            dtau,
            single_scattering_albedo,
            asymmetric_parameter,
            reflectivity_surface,
            incoming_flux,
        )
    elif rtmode == "abs":
        # mu0 = 1.0
        # mu1 = 1.0
        mu0 = jnp.cos(60.0 / 180.0 * jnp.pi)
        mu1 = jnp.cos(60.0 / 180.0 * jnp.pi)
        Fr = art.run(dtau, surface_pressure, incoming_flux, mu0, mu1)

    std = resolution_to_gaussian_std(broadening)
    Fr_inst = sop.ipgauss(Fr, std)
    Fr_samp = sop.sampling(Fr_inst, vv, nus_obs)
    return factor * Fr_samp


def cost_function(params):
    return jnp.sum((spectra - spectral_model(params)) ** 2)


if opt_perform:
    adam = OptaxSolver(opt=optax.adam(1.0e-2), fun=cost_function)
    # res = adam.run(parinit)

    params = parinit
    state = adam.init_state(params)
    val = []
    loss = []
    for _ in tqdm.tqdm(range(5000)):
        params, state = adam.update(params, state)
        val.append(params)
        loss.append(state.value)
    val = np.array(val)
    loss = np.array(loss)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(loss)
    plt.yscale("log")
    plt.show()

    # res.params
    print("fsed, sigmag, Kzz, vrv, vr, _broadening, const_mmr_ch4, factor")
    print("init:", unpack_params(parinit))
    print("best:", unpack_params(params))

    print("fsed, sigmag, Kzz, vrv, vr, _broadening, const_mmr_ch4, factor")
    print("best (packed):", params)

    F_samp = spectral_model(params)
    F_samp_init = spectral_model(parinit)

    plt = plotjupiter.plot_opt(nus_obs, spectra, F_samp_init, F_samp)
    plt.savefig("fitting.png")
    import sys

    sys.exit()

if check_atm:
    params_check = jnp.array(
        [0.8533299, 2.0, 4.0, -5.0, -58.81297833, 2.5, 1.93828069, 0.54328201]
    )
    vv, factor, broadening, asymmetric_parameter, single_scattering_albedo, dtau = (
        atmospheric_model(params_check)
    )

    import sys

    sys.exit()


# %%

# T0, log_fsed, log_Kzz, vrv, vv, boradening, mmr, normalization factor
# parinit = jnp.array(
#    [1.5, np.log10(1.0) * 10000.0, np.log10(1.0e4) * 10.0, -5.0, -55.0, 2.5, 1.2, 0.6]
# )
# multiple_factor = jnp.array([100.0, 0.0001, 0.1, 1.0, 1.0, 10000.0, 0.01, 1.0])

Pjupiter = 9.925 * 3600  # Jupiter siderial period sec
cm2km = 1.0e-5
vrotmax = (
    2 * np.pi * RJ / Pjupiter * cm2km
)  # rotation velocity of Jupiter at the equator in km/s (12.57 km/s)


def model_c(nu1, y1):

    # T0_n = numpyro.sample("T0_n", dist.Uniform(0.5, 2.0))
    log_fsed_n = numpyro.sample("log_fsed_n", dist.Uniform(0.0, 2.0))
    numpyro.deterministic("fsed", 10**log_fsed_n)
    log_Kzz_n_fixed = jnp.log10(Kzz_fixed)
    # vrv = numpyro.sample("vrv", dist.Uniform(-vrotmax, vrotmax))
    # vrv = numpyro.sample("vrv",  dist.TruncatedNormal(loc=0.0, scale=vrotmax/3.0, low=-vrotmax, high=vrotmax))
    vrv = 0.0  # fix
    vr = numpyro.sample("vr", dist.Uniform(-70.0, -40.0))
    broadening = 25000.0  # fix
    log_molmass_ch4_n = numpyro.sample("log_MMR_CH4", dist.Uniform(-1, 1))
    molmass_ch4_n = 10**log_molmass_ch4_n
    numpyro.deterministic("mmr_ch4", molmass_ch4_n * 0.01)
    factor = numpyro.sample("factor", dist.Uniform(0.0, 1.0))

    # log_fsed, sigmag, log_Kzz, vrv, vv, boradening, mmr, normalization factor
    # parinit = jnp.array(
    #    [jnp.log10(3.0), sigmag_fixed, jnp.log10(Kzz_fixed), -5.0, -55.0, 2.5, 0.2, 0.6]
    # )

    params = jnp.array(
        [
            log_fsed_n,
            sigmag_fixed,
            log_Kzz_n_fixed,
            vrv,
            vr,
            broadening,
            molmass_ch4_n,
            factor,
        ]
    )
    mean = spectral_model(params)

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    err_all = jnp.ones_like(nu1) * sigma
    # err_all = jnp.sqrt(y1err**2. + sig**2.)
    numpyro.sample("y1", dist.Normal(mean, err_all), obs=y1)


# initialization
import jax.numpy as jnp

if use_init:
    # log_fsed, sigmag, , log_Kzz, vrv, vv, boradening (fix), mmr, normalization factor
    # best (packed): [  0.79194082   2.           4.          -2.52860584 -57.54213557 2.5          0.41915007   0.54551278] #before #521
    #    best (packed): [  1.24906576   2.           4.          -3.03095345 -57.8439118
    #   2.5          0.84884493   0.53314691]

    init_params = {
        "log_fsed_n": 1.24906576,
        # "vrv": -3.03095345,
        "vr": -57.8439118,
        "log_MMR_CH4": np.log10(0.84884493),
        "factor": 0.53314691,
        "sigma": 1.0,
    }

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

if hmc_perform:
    print("HMC starts")
    num_warmup, num_samples = 500, 1000
    ######
    # num_warmup, num_samples = 10, 20
    ######
    # kernel = NUTS(model_c,forward_mode_differentiation=True)
    kernel = NUTS(model_c)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    if use_init:
        mcmc.run(rng_key_, nu1=nus_obs, y1=spectra, init_params=init_params)
    else:
        mcmc.run(rng_key_, nu1=nus_obs, y1=spectra)
    mcmc.print_summary()

    # save the samples to the netcdf file
    arviz.from_numpyro(mcmc).to_netcdf("output/samples.nc")

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
    np.savez("output/hpdi.npz", hpdi_mu1)
    np.savez("output/median.npz", median_mu1)
    np.savez("output/predictions.npz", predictions)

#####################################################
