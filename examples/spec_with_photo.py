"""
NUTS analysis of the high-resolution emission spectrum with photometric information
====================================================================================

This script performs a Bayesian analysis of the high-resolution emission spectrum with photometric information.

"""

import os

# from jax import config
# config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from exojax.spec.unitconvert import wav2nu
from exojax.spec.specop import SopPhoto
from exojax.test.emulate_spec import sample_emission_spectrum
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std

from jax import jit

# ----
# loads test spectrum data, used in Paper I (Kawahara et al. 2022)

nu_grid_obs, f_obs, f_obs_err, Kmag, Kmag_err, filter_id = sample_emission_spectrum()

# plt.plot(nu_grid_obs, f_obs, ".")
# plt.savefig("obsspec.png")

# ----
# prior knowledge of the system
distance = 1.996  # pc Bedin et al. 2023

# ----
# instrument settings

Rinst = 100000.0  # instrumental spectral resolution
beta_inst = resolution_to_gaussian_std(Rinst)

mols = ["H2O", "CH4", "CO"]
db = ["ExoMol", "HITEMP", "ExoMol"]

# ----
# normalization (just to use the spectrum to be around 1)
Fref = 2.0e-15

# ----
# wavenumber grid setting for the forward model
nu_min = np.min(nu_grid_obs)
nu_max = np.max(nu_grid_obs)

margin = 1.0
ngrid = 6000
nu_grid_spec, wav, res = wavenumber_grid(
    nu_min - margin, nu_max + margin, ngrid, xsmode="premodit"
)
print("resolution = ", res)

# ----
# photometry forward model setting
sop_photo = SopPhoto(filter_id, up_resolution_factor=10)
nu_grid_photo = sop_photo.nu_grid_filter

print("len(nu_grid_photo) = ", len(nu_grid_photo)) #11082
print("len(nu_grid_spec) = ", len(nu_grid_spec))   #6000

# photometry model is more time-consuming than the spectroscopy model

# ----
# molecules/CIA database settings, uses nu_photo becuase it's wider than nu_grid_obs
from exojax.spec.api import MdbExomol
from exojax.spec.api import MdbHitemp
from exojax.spec import contdb
from exojax.spec import molinfo

mdb_h2o = MdbExomol(
    ".database/H2O/1H2-16O/POKAZATEL", nurange=nu_grid_photo, gpu_transfer=False
)
mdb_co = MdbExomol(
    ".database/CO/12C-16O/Li2015", nurange=nu_grid_photo, gpu_transfer=False
)
mdb_ch4 = MdbHitemp(".database/CH4", nurange=nu_grid_photo, gpu_transfer=False)

molmasses = jnp.array([mdb_h2o.molmass, mdb_ch4.molmass, mdb_co.molmass])
molmasses = jnp.array([mdb_h2o.molmass])


cdbH2H2 = contdb.CdbCIA(".database/H2-H2_2011.cia", nu_grid_photo)
cdbH2He = contdb.CdbCIA(".database/H2-He_2011.cia", nu_grid_photo)

molmassH2 = molinfo.molmass("H2")
molmassHe = molinfo.molmass("He", db_HIT=False)


def mean_molecular_weight(vmr, vmrH2, vmrHe):
    mmw = jnp.sum(vmr * molmasses) + vmrH2 * molmassH2 + vmrHe * molmassHe
    return mmw


# ----
# Sets opacity calculators for spectroscopy
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opacont import OpaCIA

trange = [500.0, 2500.0]
dgres = 1.0

opa_spec_h2o = OpaPremodit(
    mdb_h2o,
    nu_grid_spec,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_spec_co = OpaPremodit(
    mdb_co,
    nu_grid_spec,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_spec_ch4 = OpaPremodit(
    mdb_ch4,
    nu_grid_spec,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_spec_cia_H2H2 = OpaCIA(cdbH2H2, nu_grid_spec)
opa_spec_cia_H2He = OpaCIA(cdbH2He, nu_grid_spec)

# ----
# Sets opacity calculators for photometry
opa_photo_h2o = OpaPremodit(
    mdb_h2o,
    nu_grid_photo,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_photo_co = OpaPremodit(
    mdb_co,
    nu_grid_photo,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_photo_ch4 = OpaPremodit(
    mdb_ch4,
    nu_grid_photo,
    auto_trange=trange,
    allow_32bit=True,
    dit_grid_resolution=dgres,
)
opa_photo_cia_H2H2 = OpaCIA(cdbH2H2, nu_grid_photo)
opa_photo_cia_H2He = OpaCIA(cdbH2He, nu_grid_photo)

# sets atmospheric radiative transfer model
from exojax.spec.atmrt import ArtEmisPure

art = ArtEmisPure(pressure_btm=1.0e2, pressure_top=1.0e-4, nlayer=200)
# does not set nu_grid because we use two types of nu_grid (for obs and photo)

# Spectral Operators (planet rotation and instrumental profile)
from exojax.spec.specop import SopRotation
from exojax.spec.specop import SopInstProfile

sop_rot = SopRotation(nu_grid_spec, vsini_max=100.0)
sop_inst = SopInstProfile(nu_grid_spec, vrmax=100.0)


from exojax.utils.astrofunc import square_radius_from_mass_logg
from exojax.utils.constants import RJ
from exojax.utils.constants import pc


# ----
# calculates the atmosphere
def calc_atmosphere(T0, alpha, logg, Mp, logvmr):
    Tarr = art.powerlaw_temperature(T0, alpha)
    Parr = art.pressure
    gravity = 10**logg
    Rp2 = square_radius_from_mass_logg(Mp, logg)

    # VMRs and mean molecular weight
    vmr = jnp.power(10.0, jnp.array(logvmr))
    vmrH2 = (1.0 - jnp.sum(vmr)) * 6.0 / 7.0
    vmrHe = (1.0 - jnp.sum(vmr)) * 1.0 / 7.0
    mmw = mean_molecular_weight(vmr, vmrH2, vmrHe)
    return Tarr, Parr, gravity, Rp2, vmr, vmrH2, vmrHe, mmw


# ----
# defines a constant vmr profile
def constant_vmr_profile(vmr):
    vmr_profile_h2o = art.constant_profile(vmr[0])
    vmr_profile_co = art.constant_profile(vmr[1])
    vmr_profile_ch4 = art.constant_profile(vmr[2])
    return vmr_profile_h2o, vmr_profile_co, vmr_profile_ch4


# ----
# defines a spectral forward model
@jit
def fspec(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini):

    Tarr, Parr, gravity, Rp2, vmr, vmrH2, vmrHe, mmw = calc_atmosphere(
        T0, alpha, logg, Mp, logvmr
    )
    vmr_profile_h2o, vmr_profile_co, vmr_profile_ch4 = constant_vmr_profile(vmr)

    # cross sections for molecules
    xsm_h2o = opa_spec_h2o.xsmatrix(Tarr, Parr)
    xsm_co = opa_spec_co.xsmatrix(Tarr, Parr)
    xsm_ch4 = opa_spec_ch4.xsmatrix(Tarr, Parr)

    # sum of the opacity for molecules
    dtaum = art.opacity_profile_xs(xsm_h2o, vmr_profile_h2o, mmw, gravity)
    dtaum = dtaum + art.opacity_profile_xs(xsm_co, vmr_profile_co, mmw, gravity)
    dtaum = dtaum + art.opacity_profile_xs(xsm_ch4, vmr_profile_ch4, mmw, gravity)

    # CIAs
    dtauH2H2 = art.opacity_profile_cia(
        opa_spec_cia_H2H2.logacia_matrix(Tarr), Tarr, vmrH2, vmrH2, mmw, gravity
    )
    dtauH2He = art.opacity_profile_cia(
        opa_spec_cia_H2He.logacia_matrix(Tarr), Tarr, vmrH2, vmrHe, mmw, gravity
    )

    # sum of the opacity
    dtau_spec = dtaum + dtauH2H2 + dtauH2He

    F0 = art.run(dtau_spec, Tarr, nu_grid=nu_grid_spec)
    Frot = sop_rot.rigid_rotation(F0, vsini, u1, u2)
    Frot_inst = sop_inst.ipgauss(Frot, beta_inst)

    mu = sop_inst.sampling(Frot_inst, RV, nu_grid_obs[0, :])
    mu = mu * (Rp2 / distance**2) * (RJ / pc) ** 2
    return mu / Fref


# ----
# defines photometry model
@jit
def fphoto(T0, alpha, logg, Mp, logvmr):
    Tarr, Parr, gravity, Rp2, vmr, vmrH2, vmrHe, mmw = calc_atmosphere(
        T0, alpha, logg, Mp, logvmr
    )
    vmr_profile_h2o, vmr_profile_co, vmr_profile_ch4 = constant_vmr_profile(vmr)

    # cross sections for molecules
    xsm_h2o = opa_photo_h2o.xsmatrix(Tarr, Parr)
    xsm_co = opa_photo_co.xsmatrix(Tarr, Parr)
    xsm_ch4 = opa_photo_ch4.xsmatrix(Tarr, Parr)

    # sum of the opacity for molecules
    dtaum = art.opacity_profile_xs(xsm_h2o, vmr_profile_h2o, mmw, gravity)
    dtaum = dtaum + art.opacity_profile_xs(xsm_co, vmr_profile_co, mmw, gravity)
    dtaum = dtaum + art.opacity_profile_xs(xsm_ch4, vmr_profile_ch4, mmw, gravity)

    # CIAs
    dtauH2H2 = art.opacity_profile_cia(
        opa_photo_cia_H2H2.logacia_matrix(Tarr), Tarr, vmrH2, vmrH2, mmw, gravity
    )
    dtauH2He = art.opacity_profile_cia(
        opa_photo_cia_H2He.logacia_matrix(Tarr), Tarr, vmrH2, vmrHe, mmw, gravity
    )

    # sum of the opacity
    dtau_photo = dtaum + dtauH2H2 + dtauH2He

    F0 = art.run(dtau_photo, Tarr, nu_grid=nu_grid_photo)
    Fobs = F0 * (Rp2 / distance**2) * (RJ / pc) ** 2

    mag = sop_photo.apparent_magnitude(Fobs)
    return mag


# ----
# checks the forward models

# examples
Mp = 33.2
RV = 28.1
T0 = 1293.0
alpha = 0.098
logvmr_sample = [-3.06, -2.83, -7.65]
logg = 4.9
vsini = 16.2

spec_model = fspec(
    T0=T0,
    alpha=alpha,
    logg=logg,
    Mp=Mp,
    logvmr=logvmr_sample,
    u1=0.0,
    u2=0.0,
    RV=RV,
    vsini=vsini,
)
print(spec_model)
mu = spec_model
err_all = f_obs_err
a = jnp.dot(mu / err_all, f_obs / err_all) / jnp.dot(mu / err_all, mu / err_all)


Kmag_model = fphoto(
    T0=T0,
    alpha=alpha,
    logg=logg,
    Mp=Mp,
    logvmr=logvmr_sample,
)
print(Kmag_model)


# fig = plt.figure(figsize=(12, 4))
fig = plt.figure(figsize=(20, 4))
plt.plot(nu_grid_obs, f_obs, ".", c="k", label="data", alpha=0.3)
plt.plot(
    nu_grid_obs,
    a * spec_model,
    label="NUTS best model (T0=1325K)",
    lw=2,
    alpha=0.7,
)
plt.ylim(0.6, 1.5)
plt.title("Kmag (model)=" + str(Kmag_model) + ", Kmag (obs)=" + str(Kmag))
plt.legend()
plt.savefig("spec.png", bbox_inches="tight")

# ----
# optimization using forward-mode differentiation
optimization = True


if optimization:
    from jax import jacfwd
    import optax

    facT = 1.0e3
    facM = 1.0e2
    facvsini = 1.0e2
    params_init = [
        1.0,
        1500.00 / facT,
        0.09,
        4.92,
        30 / facM,
        logvmr_sample,
        0.0,
        0.0,
        30.0,
        10.0,
    ]

    def objective(params):
        a, T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini = recover_params(params)
        mu = fspec(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)
        f = (a * mu - jnp.concatenate(f_obs)) / jnp.concatenate(f_obserr)
        return jnp.dot(f, f)

    def recover_params(params):
        a, T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini = params
        T0 = T0 * facT
        Mp = 33.2
        u1 = 0.0  # fix
        u2 = 0.0  # fix
        RV = 28.2  # fix
        vsini = vsini * facvsini
        return a, T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini

    # ----
    # defines the forward-mode differentiation of the objective function (JVP)
    def dfluxt_jacfwd(params):
        return jacfwd(objective)(params)

    solver = optax.adamw(learning_rate=0.001)
    opt_state = solver.init(params_init)

    trajectory = []
    import copy

    params = copy.deepcopy(params_init)

    for i in range(100):
        grad = dfluxt_jacfwd(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        trajectory.append(params)
        if np.mod(i, 10) == 0:
            print("Objective function:", params)

    a, T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini = recover_params(params)

    spec_model_opt = a * fspec(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)
    print(spec_model_opt)

    
    fig = plt.figure(figsize=(12, 4))
    plt.plot(nu_grid_obs, f_obs, ".", c="k", label="data", alpha=0.3)
    plt.plot(
        nu_grid_obs,
        spec_model_opt,
        label="optimized model",
        lw=3,
        alpha=0.8,
        color="C1",
    )
    plt.plot(
        nu_grid_obs, spec_model, label="initial model", lw=2, alpha=0.5
    )

    plt.ylim(0.7, 1.5)
    plt.legend()
    plt.savefig("spec_opt.png", bbox_inches="tight")

#


# Bayesian analysis
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

# from exojax.utils.gpkernel import gpkernel_RBF

mols_unique = ["H2O", "CO", "CH4"]


def model_c(nu_grid_obs, y1, y1err, y2, y2err):
    logg = numpyro.sample("logg", dist.Uniform(4.0, 6.0))
    # logg = numpyro.sample(
    #    "logg", dist.TruncatedNormal(5.5, 0.1, low=5.0)
    # )  # Wang et al. (2022)
    Mp = numpyro.sample("Mp", dist.Normal(33.5, 0.3))  # Brandt et al. (2021)
    # Mp = numpyro.sample(
    #    "Mp", dist.TruncatedNormal(72.7, 0.8, low=1.0)
    # )  # Brandt et al. (2021)
    RV = numpyro.sample("RV", dist.Uniform(26, 30))  # HK
    T0 = numpyro.sample("T0", dist.Uniform(1000.0, 2500.0))
    alpha = numpyro.sample("alpha", dist.Uniform(0.05, 0.15))
    vsini = numpyro.sample("vsini", dist.Uniform(10.0, 20.0))  # HK
    logvmr = []
    for i in range(len(mols_unique)):
        logvmr.append(numpyro.sample("log" + mols_unique[i], dist.Uniform(-10.0, 0.0)))
    u1 = 0.0
    u2 = 0.0

    sigma = numpyro.sample("sigma", dist.Exponential(10.0))
    sig = jnp.ones_like(nu_grid_obs) * sigma
    err_all = jnp.sqrt(y1err**2.0 + sig**2.0)

    mu = fspec(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)

    # normalizes the model spectrum to the data
    ## solution of d/da |(a mu - fobs)/err_all|^2 = 0 for a
    a = jnp.dot(mu / err_all, f_obs[0] / err_all) / jnp.dot(mu / err_all, mu / err_all)
    numpyro.deterministic("a", a)
    mu = a * mu

    # photometry
    Kmag = fphoto(T0, alpha, logg, Mp, logvmr)

    numpyro.sample("y1", dist.Normal(mu, err_all), obs=y1)
    numpyro.sample("y2", dist.Normal(Kmag, y2err), obs=y2)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
# num_warmup, num_samples = 25, 100
# num_warmup, num_samples = 50, 200
######
num_warmup, num_samples = 500, 500
######
kernel = NUTS(model_c, forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
import time

t1 = time.perf_counter()
mcmc.run(
    rng_key_,
    nu_grid_obs=nu_grid_obs[0, :],
    y1=f_obs[0],
    y1err=f_obserr[0],
    y2=Kmag,
    y2err=Ks_mag_obserr,
)
t2 = time.perf_counter()
print(t2 - t1)
mcmc.print_summary()


import arviz as az

az.plot_trace(mcmc, backend_kwargs={"constrained_layout": True})
plt.savefig("./output/trace.pdf", bbox_inches="tight")

import pickle

with open("./output_bn/mcmc.pickle", mode="wb") as f:
    pickle.dump(mcmc, f)
with open("./output_bn/samples.pickle", mode="wb") as f:
    pickle.dump(mcmc.get_samples(), f)


with open("./output_bn/samples.pickle", mode="rb") as f:
    samples = pickle.load(f)

exit()
from numpyro.diagnostics import hpdi

pred = Predictive(model_c, samples, return_sites=["y1", "y2"])
predictions = pred(
    rng_key_,
    nu_grid_obs=nu_grid_obs[0, :],
    y1=None,
    y1err=f_obserr[0, :],
    y2=None,
    y2err=Ks_mag_obserr,
)
with open("./output_bn/pred.pickle", mode="wb") as f:
    pickle.dump(predictions, f)

median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.95)
np.savez("./output_bn/all.npz", [median_mu1, hpdi_mu1[0], hpdi_mu1[1]])

median_mu2 = jnp.median(predictions["y2"], axis=0)
hpdi_mu2 = hpdi(predictions["y2"], 0.95)
np.savez("./output_bn/Kmag.npz", [median_mu2, hpdi_mu2[0], hpdi_mu2[1]])

median_mu3 = []
hpdi_mu3 = []
for i in range(len(mols_unique)):
    predictions = pred(
        rng_key_,
        nu_grid_obs=nu_grid_obs[0, :],
        y1=None,
        y1err=jnp.concatenate(f_obserr),
        y2=None,
        y2err=Ks_mag_obserr,
        onl=mols_unique[i],
    )
    median_mu3.append(jnp.median(predictions["y1"], axis=0))
    hpdi_mu3.append(hpdi(predictions["y1"], 0.95))
    np.savez(
        "./output_bn/" + mols_unique[i] + ".npz",
        [median_mu3[i], hpdi_mu3[i][0], hpdi_mu3[i][1]],
    )


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

plt.switch_backend("agg")
i_min = 0
i_max = 0
for k in range(len(ord_list)):
    i_max = i_max + len(ld_obs[k])
    print(i_min, i_max, np.shape(ld_obs))
    fig, ax = plt.subplots(figsize=(28, 6.0))
    ax.plot(ld_obs[k], f_obs[k], "+", color="black", label="data")
    ax.plot(ld_obs[k], median_mu1[i_min:i_max], color="C0", label="median")
    ax.fill_between(
        ld_obs[k],
        hpdi_mu1[0][i_min:i_max],
        hpdi_mu1[1][i_min:i_max],
        alpha=0.3,
        interpolate=True,
        color="C0",
        label="95% area",
    )

    plt.xlabel("Wavelength [$\AA$]", fontsize=15)
    plt.ylabel("Normalized Flux", fontsize=15)
    ax.set_xlim(np.min(ld_obs[k]), np.max(ld_obs[k]))
    ax.set_ylim(0.6, 1.3)

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.show()

    num = str(ord_list[k][0]) + "-" + str(ord_list[k][1])
    plt.savefig("./output/fit" + num + ".pdf", bbox_inches="tight")
    plt.close()

    i_min = i_max


i_min = 0
i_max = 0
for k in range(len(ord_list)):
    i_max = i_max + len(ld_obs[k])
    print(i_min, i_max, np.shape(ld_obs))
    fig, ax = plt.subplots(figsize=(28, 6.0))
    ax.plot(ld_obs[k], f_obs[k], "+", color="black", label="data")
    ax.plot(ld_obs[k], median_mu1[i_min:i_max], color="C0", label="all", zorder=10.0)
    ax.fill_between(
        ld_obs[k],
        hpdi_mu1[0][i_min:i_max],
        hpdi_mu1[1][i_min:i_max],
        alpha=0.3,
        interpolate=True,
        color="C0",
        label="95% area",
    )
    for i in range(len(mols_unique)):
        j = i + 1
        ax.plot(
            ld_obs[k],
            median_mu3[i][i_min:i_max],
            color="C" + str(j),
            label="w/o " + mols_unique[i],
        )

    plt.xlabel("Wavelength [$\AA$]", fontsize=15)
    plt.ylabel("Normalized Flux", fontsize=15)
    ax.set_xlim(np.min(ld_obs[k]), np.max(ld_obs[k]))
    ax.set_ylim(0.6, 1.3)

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.show()

    num = str(ord_list[k][0]) + "-" + str(ord_list[k][1])
    plt.savefig("./output/fit_wo" + num + ".pdf", bbox_inches="tight")
    plt.close()

    i_min = i_max


import corner

figure = corner.corner(
    samples,
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
    color="C0",
    label_kwargs={"fontsize": 20},
    smooth=1.0,
)  # smooth parameter needed
figure.savefig("./output/corner.pdf", bbox_inches="tight")
