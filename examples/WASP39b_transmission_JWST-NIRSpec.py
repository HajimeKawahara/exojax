"""
WASP-39 b Transmission Spectrum Retrieval with ExoJAX + NumPyro
===============================================================

This example demonstrates how to retrieve the JWST NIRSpec/G395H
transmission spectrum using *ExoJAX* and *NumPyro*'s Hamiltonian
Monte-Carlo **NUTS** sampler for Bayesian inference.

See Section 7.2 of https://arxiv.org/abs/2410.06900 for details.

Shotaro Tada, December 11st (2025)

"""
# %%

import os

import jax
from jax import random
import jax.numpy as jnp

import numpy as np
from astropy.io import fits
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import corner

from exojax.rt import ArtTransPure
from exojax.utils.constants import RJ, Rs, MJ
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.grids import wavenumber_grid, wav2nu

from exojax.postproc.specop import SopRotation, SopInstProfile

from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database import molinfo
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa

# --- Probabilistic Programming imports -------------------------------------
from numpyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro import handlers
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList
from numpyro.infer.initialization import init_to_value

# sphinx_gallery_thumbnail_path = '_static/transit.png'


# %%
# Setup and configuration
# -------------------------------------------------------------------

# Planet–star system parameters and orbital period (days)
period_day = 4.05528
Mp_mean, Mp_std = 0.281, 0.032  # [M_J]
Rstar_mean, Rstar_std = 0.939, 0.022  # [R_Sun]

# Output directory
DIR_SAVE = "output_wasp39b"
os.makedirs(DIR_SAVE, exist_ok=True)

# Opacity loading flag: set to True to load precomputed opacities
opa_load = True
# Opacity saving flag: set to True to save computed opacities
opa_save = False


# %%
# Load observed transmission spectrum
# -----------------------------------

# We use the 2.8–5.1 μm radius-ratio spectrum (``R_p/R_s``) of the
# hot-Saturn exoplanet **WASP-39 b** from the JWST Early-Release Science
# (ERS) program (Alderson et al. 2023).
wav_obs = np.load(
    "WASP39b_NIRSpec_data/wavelength.npy"
)  # observed wavelength grid (nm)
rp_mean = np.load(
    "WASP39b_NIRSpec_data/wasp39b_nirspec_g395h_rp_mean.npy"
)  # mean R_p/R_s spectrum
rp_std = np.load(
    "WASP39b_NIRSpec_data/wasp39b_nirspec_g395h_rp_std.npy"
)  # 1 std uncertainty

# Convert from wavelength to wavenumber for modelling
inst_nus = wav2nu(wav_obs, "nm")


# %%
# Instrumental resolution
# -------------------------
#
# Read the NIRSpec/G395H resolving-power curve and interpolate it so the
# forward model can convert to a Gaussian instrumental broadening kernel.


def load_resolution_curve():
    """Load and cache the NIRSpec/G395H resolution curve from the FITS table."""
    with fits.open("WASP39b_NIRSpec_data/jwst_nirspec_g395h_disp.fits") as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


_res_curve = load_resolution_curve()


def res_G395H(wavelength_nm: float) -> float:
    """Return the resolving power *R* of JWST NIRSpec/G395H at *wavelength_nm*."""
    return np.interp(wavelength_nm / 1000.0, _res_curve[:, 0], _res_curve[:, 2])


Rinst = res_G395H(np.mean(wav_obs))


# %%
# Wavenumber grid and spectral operators
# -------------------------------------------------
#
# Build a high-resolution wavenumber grid for forward modelling and construct
# spectral operators to mimic rotation and the NIRSpec line-spread function.

N = 30_000  # spectral points; lower for faster demo
nu_grid, wav_grid, res_high = wavenumber_grid(
    np.min(wav_obs) - 15, np.max(wav_obs) + 15, N=N, unit="nm", xsmode="premodit"
)
print(f"wavenumber grid: R≈{res_high:.0f}")

beta_inst = resolution_to_gaussian_std(Rinst)
sop_rot = SopRotation(nu_grid, vsini_max=100.0)  # rigid rotation kernel
sop_inst = SopInstProfile(nu_grid, vrmax=300.0)  # IP & sampling

# %%
# Atmospheric radiative‑transfer object
# -------------------------------------

diffmode = 0
nlayer = 120  # number of layers in the atmosphere
pressure_top, pressure_btm = 1e-11, 1e1  # [bar]
art = ArtTransPure(pressure_top=pressure_top, pressure_btm=pressure_btm, nlayer=nlayer)
Tlow, Thigh = 500.0, 2000.0
art.change_temperature_range(Tlow, Thigh)

# %%
# Opacity sources
# ---------------
#
# Load collision-induced absorption (CIA) and line opacities. The script
# prefers saved preMODIT snapshots (``opa_*.zarr``); if missing and
# ``opa_load=False``, it will build them from HITEMP/ExoMol databases and save
# them for reuse.

# Collision‑induced absorption (CIA)
ciapath_list = {
    "H2H2": "path_to/.db_CIA/H2-H2_2011.cia",
    "H2He": "path_to/.db_CIA/H2-He_2011.cia",
}
opa_cias = {
    name: OpaCIA(CdbCIA(path, nurange=nu_grid), nu_grid=nu_grid)
    for name, path in ciapath_list.items()
}

# Line absorption (HITEMP + ExoMol)
db_HITEMP = "path_to/.db_HITEMP/"
db_ExoMol = "path_to/.db_ExoMol/"

molpath_list_HITEMP = {
    "H2O": f"{db_HITEMP}H2O/",
    "CO": f"{db_HITEMP}CO/",
    "CO2": f"{db_HITEMP}CO2/",
    # "CH4": f"{db_HITEMP}CH4/",
}

molpath_list_Exomol = {
    # "NH3": f"{db_ExoMol}NH3/14N-1H3/CoYuTe/",
    "H2S": f"{db_ExoMol}H2S/1H2-32S/AYT2/",
    # "HCN": f"{db_ExoMol}HCN/1H-12C-14N/Harris/",
    # "C2H2": f"{db_ExoMol}C2H2/12C2-1H2/aCeTY/",
    "SO2": f"{db_ExoMol}SO2/32S-16O2/ExoAmes/",
    "SiO": f"{db_ExoMol}SiO/28Si-16O/SiOUVenIR/",
}

ndiv = 6  # preMODIT stitch blocks


def build_premodit_from_snapshot(snapshot, molmass, mol):
    """Create preMODIT opacity and persist it for reuse."""
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid,
        nstitch=ndiv,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=1,
        allow_32bit=True,
        cutwing=1 / (2 * ndiv),
    )
    saveopa(opa, "opa_" + mol + ".zarr", format="zarr", aux={"molmass": molmass})
    return opa


def load_or_build_opacity(mol, path, mdb_factory):
    """Load saved opacity or build from database snapshot."""
    if opa_load:
        opa = OpaPremodit.from_saved_opa("opa_" + mol + ".zarr", strict=False)
        return opa, opa.aux["molmass"]

    mdb = mdb_factory(path)
    molmass = mdb.molmass
    opa = build_premodit_from_snapshot(mdb.to_snapshot(), molmass, mol)
    del mdb
    return opa, molmass


def load_molecular_opacities():
    """Load or create all molecular opacities for HITEMP and ExoMol."""
    opa_mols_local = {}
    molmass_list = []

    print("Loading HITEMP/ExoMol databases …")
    for mol, path in molpath_list_HITEMP.items():
        print(f"  * {mol} (HITEMP)")
        mdb_factory = lambda p: MdbHitemp(p, nu_grid, gpu_transfer=False, isotope=1)
        opa, molmass = load_or_build_opacity(mol, path, mdb_factory)
        opa_mols_local[mol] = opa
        molmass_list.append(molmass)

    for mol, path in molpath_list_Exomol.items():
        print(f"  * {mol} (ExoMol)")
        mdb_factory = lambda p: MdbExomol(p, nu_grid, gpu_transfer=False)
        opa, molmass = load_or_build_opacity(mol, path, mdb_factory)
        opa_mols_local[mol] = opa
        molmass_list.append(molmass)

    return opa_mols_local, jnp.array(molmass_list)


opa_mols, molmass_arr = load_molecular_opacities()


# %%
# Probabilistic model
# -------------------
#
# The NumPyro model couples planetary/stellar parameters, molecular mixing
# ratios, a grey cloud deck, and a simple isothermal temperature structure. It
# produces a model transmission spectrum convolved with rotation and the
# instrumental profile, then compares it to the observed ``R_p/R_s`` data.


def model_c(rp_mean, rp_std):
    """NumPyro model: forward spectral model + priors."""

    # --- Planet / star parameters -----------------------------------------
    Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0)) * MJ
    Rstar = (
        numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0)) * Rs
    )
    radius_btm = numpyro.sample("Radius_btm", dist.Uniform(1.0, 1.5)) * RJ
    RV = numpyro.sample("RV", dist.Uniform(-200, 0))

    # --- Atmospheric composition -----------------------------------------
    vmr_arr = []
    for mol in opa_mols:
        logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(-15, 0))
        vmr_arr.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
    vmr_arr = jnp.array(vmr_arr)

    vmr_tot = jnp.clip(jnp.sum(vmr_arr, axis=0), 0.0, 1.0)
    vmrH2 = (1.0 - vmr_tot) * 6.0 / 7.0
    vmrHe = (1.0 - vmr_tot) * 1.0 / 7.0

    mmw = (
        molinfo.molmass_isotope("H2") * vmrH2
        + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
        + jnp.dot(molmass_arr, vmr_arr)
    )

    # --- Temperature structure -------------------------------------------
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    Tarr = T0 * jnp.ones_like(art.pressure)  # constant T profile

    # --- Grey cloud deck ---------------------------------------------------
    # We model a wavelength-independent (gray) cloud deck as a Gaussian in
    # log10-pressure. The cloud center logP_cloud is a free parameter.
    logP_cloud = numpyro.sample("logP_cloud", dist.Uniform(-11, 1))

    # Fixed cloud width in log10(P) space (narrow deck).
    width_cloud = 1.0 / 25.0

    # Set the Gaussian amplitude so that the *integrated* cloud optical depth
    # over the atmosphere is ~50, independent of the number of layers.
    dtau_c = (
        50.0
        * ((jnp.log10(pressure_btm) - jnp.log10(pressure_top)) / nlayer)
        / width_cloud
    )
    pressure_arr = jnp.log10(art.pressure)
    cloud_profile = (pressure_arr[:, None] - logP_cloud) / width_cloud
    # Per-layer optical-depth increment: normalized Gaussian in log10(P).
    dtau_cloud = (
        dtau_c / jnp.sqrt(jnp.pi) * jnp.exp(-jnp.clip(cloud_profile**2, -50, 50))
    )
    # Broadcast to all wavelengths to make the cloud gray.
    dtau_cloud = jnp.broadcast_to(dtau_cloud, (pressure_arr.size, nu_grid.size))

    # --- Gravity profile --------------------------------------------------
    gravity_btm = gravity_jupiter(radius_btm / RJ, Mp / MJ)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    # --- Opacity summation -------------------------------------------------
    dtau = dtau_cloud

    # CIA
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        logacia_matrix = opa_cias[molA + molB].logacia_matrix(Tarr)
        vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
        dtau += art.opacity_profile_cia(
            logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], gravity
        )

    # Line opacity
    for i, mol in enumerate(opa_mols):
        xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
        dtau += art.opacity_profile_xs(xsmatrix, vmr_arr[i], mmw[:, None], gravity)

    # --- Radiative‑transfer ------------------------------------------------
    rp2 = art.run(
        dtau, Tarr, mmw, radius_btm, gravity_btm
    )  # (radius/radius_btm)^2 spectrum

    # --- Broadening kernels ------------------------------------------------
    vsini = 2 * jnp.pi * radius_btm / (period_day * 86400) / 1e5  # km/s
    u1 = u2 = 0.0  # quadratic limb‑darkening (planet)
    Frot = sop_rot.rigid_rotation(rp2, vsini, u1, u2)
    Frot_inst = sop_inst.ipgauss(Frot, beta_inst)
    Rp2_sample = sop_inst.sampling(Frot_inst, RV, inst_nus)

    mu = jnp.sqrt(Rp2_sample) * (radius_btm / Rstar)  # (radius/Rstar) spectrum

    # --- Likelihood -------------------------------------------------------
    numpyro.deterministic("rp_mu", mu[::-1])
    numpyro.sample("rp", dist.Normal(mu[::-1], rp_std), obs=rp_mean)


# %%
# Stochastic Variational Inference (SVI) warm-up for HMC-NUTS (Optional)
# ----------------------------------------------------------------------
#
# Run stochastic variational inference with a custom guide that keeps Mp and
# Rs on their priors while fitting an AutoMultivariateNormal to the remaining
# latent variables. The SVI median seeds HMC and its Fisher information is
# reused as a mass matrix estimate.


def prior_guide(rp_mean, rp_std):
    """Guide for Mp and Rs so they follow their priors during SVI."""
    Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0))
    Rs = numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0))
    return {"Mp": Mp, "Rs": Rs}


def build_guide():
    """Construct guide with separated priors for Mp/Rs and AutoMVN for the rest."""
    guide = AutoGuideList(model_c)
    guide.append(prior_guide)
    # Hide rp_mu so the Auto guide only sees latent sample sites
    model_hidden = handlers.block(model_c, hide=["Mp", "Rs", "rp_mu"])
    guide.append(AutoMultivariateNormal(model_hidden))
    return guide


def save_svi_outputs(params, losses, init_values, output_dir):
    """Persist SVI artifacts for reuse or inspection."""
    params_cpu = {k: np.asarray(jax.device_get(v)) for k, v in params.items()}
    losses_cpu = np.asarray(jax.device_get(losses))
    init_cpu = {k: np.asarray(jax.device_get(v)) for k, v in init_values.items()}

    np.savez(os.path.join(output_dir, "svi_params.npz"), **params_cpu)
    np.save(os.path.join(output_dir, "svi_losses.npy"), losses_cpu)
    np.savez(os.path.join(output_dir, "svi_init_values.npz"), **init_cpu)

    print(f"SVI params saved to {output_dir}/svi_params.npz")
    print(f"SVI losses saved to {output_dir}/svi_losses.npy")
    print(f"SVI init values saved to {output_dir}/svi_init_values.npz")


def run_svi(rng_key, rp_mean, rp_std, num_steps=1000, lr=0.005):
    """Execute SVI, return params, losses, init strategy, median, and guide."""
    guide = build_guide()
    optimizer = optim.Adam(lr)
    svi = SVI(model_c, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key,
        num_steps,
        rp_mean=rp_mean,
        rp_std=rp_std,
    )

    params = svi_result.params
    losses = svi_result.losses

    # Median of the AutoMVN part in the constrained space.
    svi_median = guide[-1].median(params)
    # Keep Mp and Rs anchored to their prior means for HMC initialisation.
    svi_median.update({"Mp": Mp_mean, "Rs": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, DIR_SAVE)
    return params, losses, init_strategy, svi_median, guide


rng_key = random.PRNGKey(0)

print("Stochastic Variational Inference (SVI) to find initial values for HMC-NUTS …")

rng_key, rng_key_ = random.split(rng_key)

_svi_params, losses, init_strategy, svi_median, svi_guide = run_svi(
    rng_key_, rp_mean=rp_mean, rp_std=rp_std
)
print(f"Final SVI loss: {float(losses[-1]):.2f}")
print("HMC initial values:", init_strategy)

# %%
# HMC-NUTS sampling
# -----------------

print("Launching HMC-NUTS …")

kernel = NUTS(
    model_c,
    max_tree_depth=5,
    init_strategy=init_strategy,
)
num_warmup, num_samples = 1000, 1000
rng_key, rng_key_ = random.split(rng_key)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, rp_mean=rp_mean, rp_std=rp_std)

# Print summary to console *and* save to file
mcmc.print_summary()
with open(os.path.join(DIR_SAVE, "mcmc_summary.txt"), "w") as f:
    with redirect_stdout(f):
        mcmc.print_summary()

# Save posterior samples and predictive spectra
posterior_sample = mcmc.get_samples()
jnp.savez(os.path.join(DIR_SAVE, "posterior_sample"), **posterior_sample)

print("Generating predictive spectrum …")

pred = Predictive(model_c, posterior_sample, return_sites=["rp"])
predictions = pred(rng_key_, rp_mean=None, rp_std=rp_std)

jnp.save(os.path.join(DIR_SAVE, "rp_pred"), predictions["rp"])


# %%
# Plotting
# --------
#
# Generate quick-look diagnostics: SVI loss curve, HMC predictive spectrum,
# observed/SVI/HMC overlay, and corner plots for a subset of parameters.
print("Plotting SVI and HMC diagnostics …")


def plot_svi_loss(loss_values, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(loss_values))
    ax.plot(x, np.asarray(loss_values), lw=1.5)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Loss")
    ax.set_title("SVI loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_overlay(wavelength_nm, rp_obs, rp_err, rp_hmc, rp_svi, save_path):
    rp_hmc_np = np.asarray(rp_hmc)
    mean = rp_hmc_np.mean(axis=0)
    std = rp_hmc_np.std(axis=0)
    rp_svi_np = np.asarray(rp_svi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        wavelength_nm,
        rp_obs,
        yerr=rp_err,
        fmt=".",
        ms=1,
        color="k",
        ecolor="0.3",
        elinewidth=0.5,
        alpha=0.5,
        label="Observed",
    )
    ax.fill_between(
        wavelength_nm,
        mean - std,
        mean + std,
        color="C0",
        alpha=0.25,
        label="HMC ±1$\sigma$",
    )
    ax.plot(wavelength_nm, mean, color="C0", lw=1.4, label="HMC mean")
    ax.plot(wavelength_nm, rp_svi_np, color="C3", lw=1.4, label="SVI median model")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$R_p/R_s$")
    ax.set_title("Observed vs SVI vs HMC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _corner_data(sample_dict, variables):
    cols = []
    labels = []
    available = [v for v in variables if v in sample_dict]
    for var in available:
        arr = np.asarray(sample_dict[var])
        arr = arr.reshape(arr.shape[0], -1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            labels.append(var if arr.shape[1] == 1 else f"{var}[{j}]")
    if not cols:
        return None, None
    return np.column_stack(cols), labels


def plot_corner(hmc_samples=None, svi_samples=None, variables=None, save_path=None):
    """Corner plot helper: supports HMC only, SVI only, or HMC+SVI overlay."""
    datasets = []
    labels = None

    if hmc_samples is not None:
        hmc_data, labels = _corner_data(hmc_samples, variables)
        if hmc_data is not None:
            datasets.append((hmc_data, "C0", {}))

    if svi_samples is not None:
        svi_data, labels_svi = _corner_data(svi_samples, variables)
        if labels is None:
            labels = labels_svi
        if svi_data is not None:
            datasets.append((svi_data, "C3", {"hist_kwargs": {"linestyle": "--"}}))

    if not datasets or labels is None:
        print("No data for corner plot; skipping.")
        return

    fig = None
    for data, color, extra_kwargs in datasets:
        fig = corner.corner(
            data,
            labels=labels,
            color=color,
            bins=40,
            smooth=1.0,
            fig=fig,
            show_titles=True,
            **extra_kwargs,
        )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# Generate deterministic rp_mu from SVI median parameters
rng_key, rng_plot = random.split(rng_key)
svi_pred = Predictive(model_c, params=svi_median, num_samples=1, return_sites=["rp_mu"])
svi_mu = svi_pred(rng_plot, rp_mean=rp_mean, rp_std=rp_std)["rp_mu"][0]

# Draw samples from the SVI guide for visualization
rng_key, rng_svi = random.split(rng_key)
svi_samples = svi_guide[-1].sample_posterior(
    rng_svi,
    _svi_params,
    rp_mean=rp_mean,
    rp_std=rp_std,
    sample_shape=(1000,),
)

loss_plot_path = os.path.join(DIR_SAVE, "svi_loss.png")
plot_svi_loss(losses, loss_plot_path)

overlay_plot_path = os.path.join(DIR_SAVE, "spectrum_overlay.png")
plot_overlay(wav_obs, rp_mean, rp_std, predictions["rp"], svi_mu, overlay_plot_path)

corner_vars = ["Radius_btm", "T0", "logP_cloud", "RV"]
corner_vars += [f"logVMR_{mol}" for mol in list(opa_mols.keys())]
corner_plot_path = os.path.join(DIR_SAVE, "corner_plot_svi.png")
plot_corner(svi_samples=svi_samples, variables=corner_vars, save_path=corner_plot_path)

hmc_corner_plot_path = os.path.join(DIR_SAVE, "corner_plot.png")
plot_corner(
    hmc_samples=posterior_sample, variables=corner_vars, save_path=hmc_corner_plot_path
)

hmc_svi_corner_overlay_path = os.path.join(DIR_SAVE, "corner_plot_hmc_svi_overlay.png")
plot_corner(
    hmc_samples=posterior_sample,
    svi_samples=svi_samples,
    variables=corner_vars,
    save_path=hmc_svi_corner_overlay_path,
)
