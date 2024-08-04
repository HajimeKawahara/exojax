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


from jax import config

config.update("jax_enable_x64", True)

figshow = False


# Forget about the following code. I need this to run the code somewhere...
# username="exoplanet01"
username = "kawahara"
if username == "exoplanet01":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("jupiter_corrected.dat")
unmask_nus_obs = dat[:, 0]
unmask_spectra = dat[:, 1]

mask = (
    (unmask_nus_obs < 6163.5)
    + ((unmask_nus_obs > 6166) & (unmask_nus_obs < 6184.5))
    + (unmask_nus_obs > 6186)
)
nus_obs = unmask_nus_obs[mask]
spectra = unmask_spectra[mask]
if figshow:
    fig = plt.figure(figsize=(20, 5))
    plt.plot(nus_obs, spectra, ".")
    plt.plot(unmask_nus_obs, unmask_spectra, alpha=0.5)
    plt.show()


# ## Solar spectrum

# This is the reflected "solar" spectrum by Jupiter! So, we need the solar spectrum template.
#
# I found very good one: High-resolution solar spectrum taken from Meftar et al. (2023)
#
# - 10.21413/SOLAR-HRS-DATASET.V1.1_LATMOS
# - http://doi.latmos.ipsl.fr/DOI_SOLAR_HRS.v1.1.html
# - http://bdap.ipsl.fr/voscat_en/solarspectra.html
#


from exojax.spec.unitconvert import wav2nu
import pandas as pd

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
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(211)
    plt.plot(nus_obs, spectra, label="masked spectrum")
    plt.plot(nus_solar * (1.0 + vperc), solspec, lw=1, label="Solar")
    plt.xlabel("wavenumber (cm-1)")
    plt.xlim(np.min(nus_obs), np.max(nus_obs))
    plt.ylim(0.0, 0.25)


# ## Atmospheric Model Setting

# See `Jupiter_cloud_model_using_amp.ipynb`


# set the master wavenumber grid
from exojax.utils.grids import wavenumber_grid

nus, wav, res = wavenumber_grid(
    np.min(nus_obs) - 5.0, np.max(nus_obs) + 5.0, 10000, xsmode="premodit", unit="cm-1"
)
print(res)

from exojax.spec.atmrt import ArtReflectPure
from exojax.utils.astrofunc import gravity_jupiter

art = ArtReflectPure(nu_grid=nus, pressure_btm=1.0e2, pressure_top=1.0e-3, nlayer=100)
art.change_temperature_range(80.0, 400.0)
Tarr = art.powerlaw_temperature(150.0, 0.2)
Parr = art.pressure

mu = 2.3  # mean molecular weight
gravity = gravity_jupiter(1.0, 1.0)

if figshow:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Tarr, art.pressure)
    ax.invert_yaxis()
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


from exojax.spec.pardb import PdbCloud
from exojax.atm.atmphys import AmpAmcloud

pdb_nh3 = PdbCloud("NH3")
amp_nh3 = AmpAmcloud(pdb_nh3, bkgatm="H2")
amp_nh3.check_temperature_range(Tarr)

# condensate substance density
rhoc = pdb_nh3.condensate_substance_density  # g/cc

from exojax.utils.zsol import nsol
from exojax.atm.mixratio import vmr2mmr
from exojax.spec.molinfo import molmass_isotope

n = nsol()  # solar abundance
abundance_nh3 = n["N"]
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

miegird_generate = False
if miegird_generate:
    fsed_grid = np.logspace(np.log10(fsed_range[0]), np.log10(fsed_range[1]), 3)
    Kzz_grid = np.logspace(np.log10(Kzz_range[0]), np.log10(Kzz_range[1]), 5)

    import matplotlib.pyplot as plt

    rg_val = []
    for fsed in fsed_grid:
        for Kzz in Kzz_grid:
            rg_layer, MMRc = amp_nh3.calc_ammodel(
                Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
            )
            rg_val.append(np.nanmean(rg_layer))
            plt.plot(fsed, np.nanmean(rg_layer), ".", color="black")
            plt.text(fsed, np.nanmean(rg_layer), f"{Kzz:.1e}")
    rg_val = np.array(rg_val)
    plt.yscale("log")
    plt.show()

    rg_range = [np.min(rg_val), np.max(rg_val)]
    N_rg = 10
    rg_grid = np.logspace(np.log10(rg_range[0]), np.log10(rg_range[1]), N_rg)

    # make miegrid
    pdb_nh3.generate_miegrid(
        sigmagmin=sigmag,
        sigmagmax=sigmag,
        Nsigmag=1,
        log_rg_min=np.log10(rg_range[0]),
        log_rg_max=np.log10(rg_range[1]),
        Nrg=N_rg,
    )
    print("Please rerun after setting miegird_generate = True")
    import sys

    sys.exit()
else:
    pdb_nh3.load_miegrid()


# fsed = 3.0
# rg_layer, MMRc = amp_nh3.calc_ammodel(Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3)
# rg_example = np.nanmean(rg_layer)
# pdb_nh3.miegrid_interpolated_values(rg_example, sigmag)


import matplotlib.pyplot as plt

# to convert MMR to g/L ...
from exojax.atm.idealgas import number_density
from exojax.utils.constants import m_u

fac = molmass_nh3 * m_u * number_density(Parr, Tarr) * 1.0e3  # g/L

if figshow:
    fig = plt.figure()
    ax = fig.add_subplot(131)
    plt.plot(rg_layer, Parr)
    plt.xlabel("rg (cm)")
    plt.ylabel("pressure (bar)")
    plt.yscale("log")
    ax.invert_yaxis()
    ax = fig.add_subplot(132)
    plt.plot(MMRc, Parr)
    plt.xlabel("condensate MMR")
    plt.yscale("log")
    # plt.xscale("log")
    ax.invert_yaxis()
    ax = fig.add_subplot(133)
    plt.plot(fac * MMRc, Parr)
    plt.xlabel("cloud density g/L")
    plt.yscale("log")
    # plt.xscale("log")
    ax.invert_yaxis()

# test fsed value
fsed = 3.0
rg_layer, MMRc = amp_nh3.calc_ammodel(
    Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
)
rg = np.nanmean(rg_layer)

from exojax.spec.opacont import OpaMie

opa_nh3 = OpaMie(pdb_nh3, nus)

sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector(
    rg, sigmag
)  # if using MieGrid
# sigma_extinction, sigma_scattering, asymmetric_factor = (
#    opa_nh3.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
# ) # direct computation


# plt.plot(pdb_nh3.refraction_index_wavenumber, miepar[50,:,0])
if figshow:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(311)
    plt.plot(nus, asymmetric_factor, color="black")
    plt.xscale("log")
    plt.ylabel("$g$")
    ax = fig.add_subplot(312)
    plt.plot(
        nus,
        sigma_scattering / sigma_extinction,
        label="single scattering albedo",
        color="black",
    )
    plt.xscale("log")
    plt.ylabel("$\\omega$")
    ax = fig.add_subplot(313)
    plt.plot(nus, sigma_extinction, label="ext", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("$\\beta_0$")
    plt.savefig("miefig_high.png")
    plt.show()

# Next, we consider the gas component, i.e. methane

# In[12]:


from exojax.spec.api import MdbHitemp


# Oh, we need HITEMP because our observation is in visible band, i.e., we need higher energy states.

# In[13]:


Eopt = 3300.0  # this is from the Elower optimization result
# mdb_reduced = MdbHitemp("CH4", nurange=[nus_start,nus_end], isotope=1, elower_max=Eopt)
mdb_reduced = MdbHitemp("CH4", nurange=[nus[0], nus[-1]], isotope=1, elower_max=Eopt)


# In[14]:


from exojax.spec.opacalc import OpaPremodit

opa = OpaPremodit(
    mdb_reduced, nu_grid=nus, allow_32bit=True, auto_trange=[80.0, 300.0]
)  # this reduced the device memory use in 5/6.


#dtau_cld = art.opacity_profile_cloud_lognormal(
#    sigma_extinction, rhoc, MMRc, rg, sigmag, gravity
#)


# # Spectrum Model

import jax.numpy as jnp

nusjax = jnp.array(nus)
nusjax_solar = jnp.array(nus_solar)
solspecjax = jnp.array(solspec)


# ### gas opacity

molmass_ch4 = molmass_isotope("CH4", db_HIT=False)

#dtau_cld_scat = art.opacity_profile_cloud_lognormal(
#    sigma_scattering, rhoc, MMRc, rg, sigmag, gravity
#)
#sigma_scattering[None, :] / sigma_extinction[None, :] + np.zeros(
#    (len(art.pressure), len(nus))
#)
asymmetric_parameter = asymmetric_factor + np.zeros((len(art.pressure), len(nus)))
reflectivity_surface = np.zeros(len(nus))


from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec.specop import SopInstProfile

sop = SopInstProfile(nus)

from exojax.utils.constants import c  # light speed in km/s

# T0, log_fsed, log_Kzz, vrv, vv, boradening, mmr, normalization factor
# parinit = jnp.array([1.5, np.log10(1.0)*10000.0, np.log10(1.e4)*10.0, -5.0, -55.0, 2.5, 1.0, 1.0])
parinit = jnp.array(
    [1.5, np.log10(1.0) * 10000.0, np.log10(1.0e4) * 10.0, -5.0, -55.0, 2.5, 1.2, 0.6]
)
multiple_factor = jnp.array([100.0, 0.0001, 0.1, 1.0, 1.0, 10000.0, 0.1, 1.0])


def spectral_model(params):

    T0, log_fsed, log_Kzz, vrv, vv, _broadening, const_mmr_ch4, factor = (
        params * multiple_factor
    )
    fsed = 10**log_fsed
    Kzz = 10**log_Kzz

    #fsed = 4.0  # 4 does not work.... 3 is OK...
    #Kzz = 1.0e4
    #vrv = -5.0
    #vv = -55.0
    #factor = 0.7
    broadening = 25000.0
    #const_mmr_ch4 = 0.12
    #T0 = 150.0

    # temperatures
    Tarr = art.powerlaw_temperature(T0, 0.2)

    # cloud
    rg_layer, MMRc = amp_nh3.calc_ammodel(
        Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
    )
    rg = jnp.mean(rg_layer)

    ### this one
    # sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
    sigma_extinction, sigma_scattering, asymmetric_factor = opa_nh3.mieparams_vector(
        rg, sigmag
    )
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
    #single_scattering_albedo = (sigma_scattering) / (sigma_extinction + xsmatrix) 


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


F_samp_init = spectral_model(unpack_params(parinit))


def cost_function(params):
    return jnp.sum((spectra - spectral_model(params)) ** 2)


from jaxopt import OptaxSolver
import optax

adam = OptaxSolver(opt=optax.adam(1.0e-2), fun=cost_function)
res = adam.run(parinit)
# maxiter = 100
# solver = jaxopt.LBFGS(fun=cost_function, maxiter=maxiter)
# res = solver.run(optpar_init)

# res.params
print(unpack_params(res.params))
print(unpack_params(parinit))


F_samp = spectral_model(res.params)
F_samp_init = spectral_model(parinit)

print(F_samp)
if True:
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    plt.plot(nus_obs, spectra, ".", label="observed spectrum")
    plt.plot(nus_obs, F_samp_init, alpha=0.5, label="init", color="C1", ls="dashed")
    plt.plot(nus_obs, F_samp, alpha=0.5, label="best fit", color="C1", lw=3)
    # plt.plot(nus,Fr_samp,alpha=0.5,label="forward modelling",color="C1")
    # plt.plot(nus*(1-vpercp_only_show_solar),incoming_flux, label="incoming",color="C2")
    plt.legend()
    plt.xlim(np.min(nus_obs), np.max(nus_obs))
    # plt.ylim(0.0, 0.25)
    plt.savefig("Jupiter_petitIRD.png")
    plt.show()
