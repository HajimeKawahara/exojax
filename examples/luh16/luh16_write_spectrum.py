"""
Computing real brown dwarf spectrum of Luhman16B, mimicking the analysis in Yama et al. arXiv:2511.23018
==============================================================================================================================

This script computes the brown dwarf spectrum of Luhman16B using ExoJAX, and compares real VLT/CRIRES data  from Crossfield et al. (2014) Nature, 505(7485):654–656.

"""

# %%

from jax import config
config.update("jax_enable_x64", True)

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd

from exojax.rt.emis import ArtEmisPure
from exojax.database import MdbExomol
from exojax.opacity import OpaPremodit
from exojax.database.cia.api import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.postproc.response import ipgauss_sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.database import molinfo
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.opacity import saveopa


# %% 
# Target data loading and masking
# ----------------------------------

target = "luh16B"
dat = pd.read_csv(
    "luhman16b_spectra.csv",
    delimiter=",",
    names=("wav", "flux", "error"),
    header=0,
)

wav_arr = dat["wav"].to_numpy() * 10000
idx = (22880 < wav_arr) * (wav_arr < 23500)
mask1 = (22898 > wav_arr) + (wav_arr > 22900.5)
mask2 = (23074.5 > wav_arr) + (wav_arr > 23076)
mask3 = (23285 > wav_arr) + (wav_arr > 23285.5)
mask4 = (23402.5 > wav_arr) + (wav_arr > 23404)
masked = mask1 * mask2 * mask3 * mask4 * idx

wavd = dat["wav"].values[masked] * 10000
flux = dat["flux"].values[masked]
err = dat["error"].values[masked]
nusd = jnp.array(1.0e8 / wavd)

# %% 
# wavelength grid and atmospheric setting
# -----------------------------------------------
Nx = 4500
nu_grid, wav, res = wavenumber_grid(
    np.min(wavd) - 5.0, np.max(wavd) + 5.0, Nx, unit="AA", xsmode="premodit"
)

# %% 
# Atmospheric setting by "art" 
# -----------------------------------------------
Tlow = 210.0
Thigh = 3500.0
art = ArtEmisPure(nu_grid=nu_grid, pressure_top=1.0e-4, pressure_btm=1.0e2, nlayer=101)
art.change_temperature_range(Tlow, Thigh)

# %% 
# instrumental setting
# -----------------------------------------------
Rinst = 100000.0
beta_inst = resolution_to_gaussian_std(Rinst)

# %% 
# database setting
# -----------------------------------------------

print("Use ExoMol CO line list.")
mol_paths = {
    "CO": os.path.expanduser("~/data_mol/.database/CO/12C-16O/Li2015"),
    "H2O": os.path.expanduser("~/data_mol/.database/H2O/1H2-16O/POKAZATEL"),
    "CH4": os.path.expanduser("~/data_mol/.database/CH4/12C-1H4/MM"),
    "HF": os.path.expanduser("~/data_mol/.database/HF/1H-19F/Coxon-Hajig"),
    "H2H2": os.path.expanduser("~/data_mol/.database/H2-H2_2011.cia"),
    "H2He": os.path.expanduser("~/data_mol/.database/H2-He_2011.cia"),
}  # *CO from ExoMol


# %% 
# calculates or loads OPA
# -----------------------------------------------
def calc_or_load_opa(
    molecule, nu_grid, Tlow, Thigh, elower_max, mol_paths, filename, diffmode=0
):
    if os.path.exists(filename):
        print("load from saved opa:", filename)
        opa = OpaPremodit.from_saved_opa(filename)
        molmass_ = opa.aux["molmass"]
    else:
        print("########################")
        print("no saved opa found:", filename)
        print("########################")

        mdb = MdbExomol(
            mol_paths[molecule],
            nurange=nu_grid,
            gpu_transfer=False,
            elower_max=elower_max,
        )
        molmass_ = mdb.molmass  # we use molmass later
        snap = mdb.to_snapshot()  # extract snapshot from mdb
        del mdb  # save the memory
        opa = OpaPremodit.from_snapshot(
            snap,
            nu_grid=nu_grid,
            diffmode=diffmode,
            auto_trange=[Tlow, Thigh],
            dit_grid_resolution=1.0,
        )
        saveopa(opa, filename, format="zarr", aux={"molmass": molmass_})
    return opa, molmass_


# %% 
# calculates or loads molecules
# -----------------------------------------------

opaCO, molmassCO = calc_or_load_opa(
    "CO", nu_grid, Tlow, Thigh, 58242.689, mol_paths, "opaCO.zarr", diffmode=0
)
opaH2O, molmassH2O = calc_or_load_opa(
    "H2O", nu_grid, Tlow, Thigh, 23726.625476, mol_paths, "opaH2O.zarr", diffmode=0
)
opaCH4, molmassCH4 = calc_or_load_opa(
    "CH4", nu_grid, Tlow, Thigh, 9900.0, mol_paths, "opaCH4.zarr", diffmode=0
)
opaHF, molmassHF = calc_or_load_opa(
    "HF", nu_grid, Tlow, Thigh, 20000.0, mol_paths, "opaHF.zarr", diffmode=0
)


# %% 
# CIA setting
# -----------------------------------------------

cia_h2h2_path = os.path.expanduser("~/data_mol/.database/H2-H2_2011.cia")
cdbH2H2 = CdbCIA(cia_h2h2_path, nu_grid)
opciaH2H2 = OpaCIA(cdb=cdbH2H2, nu_grid=nu_grid)


cia_h2he_path = os.path.expanduser("~/data_mol/.database/H2-He_2011.cia")
cdbH2He = CdbCIA(cia_h2he_path, nu_grid)
opciaH2He = OpaCIA(cdb=cdbH2He, nu_grid=nu_grid)


# %% 
# molecular mass setting
# -----------------------------------------------

molmassH2 = molinfo.molmass_isotope("H2")
molmassHe = molinfo.molmass_isotope("He", db_HIT=False)
M = {
    "H2": molmassH2,
    "He": molmassHe,
    "CO": molmassCO,
    "H2O": molmassH2O,
    "CH4": molmassCH4,
    "HF": molmassHF,
}


# %% 
# setting a spectrum function
# -----------------------------------------------

vsini_max = 100.0
vr_array = velocity_grid(res, vsini_max)


def frun(
    Tarr,
    MMR_CO,
    MMR_H2O,
    MMR_CH4,
    MMR_HF,
    logg,
    u1,
    u2,
    RV,
    vsini,
    mnorm1,
    mnorm2,
    mnorm3,
    mnorm4,
    logP_ctop,
    width_c,
    VMR_H2,
    VMR_He,
    mmw,
):

    # gravity
    g = 10**logg  # gravity in the unit of Jupiter

    # molecule
    xsmatrixCO = opaCO.xsmatrix(Tarr, art.pressure)
    mmr_arr_CO = art.constant_mmr_profile(MMR_CO)
    dtaumCO = art.opacity_profile_xs(xsmatrixCO, mmr_arr_CO, molmassCO, g)

    xsmatrixH2O = opaH2O.xsmatrix(Tarr, art.pressure)
    mmr_arr_H2O = art.constant_mmr_profile(MMR_H2O)
    dtaumH2O = art.opacity_profile_xs(xsmatrixH2O, mmr_arr_H2O, molmassH2O, g)
    xsmatrixCH4 = opaCH4.xsmatrix(Tarr, art.pressure)
    mmr_arr_CH4 = art.constant_mmr_profile(MMR_CH4)
    dtaumCH4 = art.opacity_profile_xs(xsmatrixCH4, mmr_arr_CH4, molmassCH4, g)

    xsmatrixHF = opaHF.xsmatrix(Tarr, art.pressure)
    mmr_arr_HF = art.constant_mmr_profile(MMR_HF)
    dtaumHF = art.opacity_profile_xs(xsmatrixHF, mmr_arr_HF, molmassHF, g)

    # *continuum
    logacia_matrixH2H2 = opciaH2H2.logacia_matrix(Tarr)
    dtaucH2H2 = art.opacity_profile_cia(
        logacia_matrixH2H2, Tarr, VMR_H2, VMR_H2, mmw, g
    )

    logacia_matrixH2He = opciaH2He.logacia_matrix(Tarr)
    dtaucH2He = art.opacity_profile_cia(
        logacia_matrixH2He, Tarr, VMR_H2, VMR_He, mmw, g
    )

    # *total tau of CIA + molcules
    dtau_m = dtaumCO + dtaumH2O + dtaumCH4 + dtaumHF + dtaucH2H2 + dtaucH2He

    # *total tau with cloud opacity
    dtau_c_norm = (
        500
        * jnp.exp(-((jnp.log10(art.pressure) - logP_ctop) ** 2) / (2 * width_c**2))
        / (jnp.sqrt(2 * jnp.pi) * width_c)
    )
    dtau_c = dtau_c_norm[:, None]
    dtau = dtau_m + dtau_c

    F = art.run(dtau, Tarr)
    Fchips = jnp.array_split(F, 4)
    F0 = jnp.concatenate(
        [
            (Fchips[0] / jnp.average(Fchips[0])) / mnorm1,
            (Fchips[1] / jnp.average(Fchips[1])) / mnorm2,
            (Fchips[2] / jnp.average(Fchips[2])) / mnorm3,
            (Fchips[3] / jnp.average(Fchips[3])) / mnorm4,
        ]
    )

    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
    mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV, vr_array)
    return mu

# %% 
# load parameters and temperature profile
# -----------------------------------------------

(
    MMR_CO0,
    MMR_H2O0,
    MMR_CH40,
    MMR_HF0,
    logg0,
    vsini0,
    mnorm10,
    mnorm20,
    mnorm30,
    mnorm40,
    logP_ctop0,
    width_c0,
    VMR_H20,
    VMR_He0,
    RV0,
    mmw0,
    u10,
    u20,
) = np.loadtxt("parameters.dat", unpack=True)

Tarr0 = np.loadtxt("tarr.dat")


# %%
# compute a single spectrum with the loaded parameters
# -----------------------------------------------

mu_single = frun(
    jnp.asarray(Tarr0),  # (nlayer,)
    float(MMR_CO0),
    float(MMR_H2O0),
    float(MMR_CH40),
    float(MMR_HF0),
    float(logg0),
    float(u10),
    float(u20),
    float(RV0),
    float(vsini0),
    float(mnorm10),
    float(mnorm20),
    float(mnorm30),
    float(mnorm40),
    float(logP_ctop0),
    float(width_c0),
    float(VMR_H20),
    float(VMR_He0),
    float(mmw0),
)

# %% 
# plotting the result
# -----------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
wav_model = 1e8 / np.asarray(nusd)
ax.plot(wav_model, flux, "+", color="gray", label="Data")
ax.plot(wav_model, np.asarray(mu_single), lw=1, label="Model", alpha=0.8, color="C1")
ax.set_xlim(wav_model.min(), wav_model.max())  
ax.set_xlabel("Wavelength [angstrom]")
ax.set_ylabel("Flux (normalized)")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("single_spectrum_sample_median.png", dpi=200)

print("saved to: single_spectrum_sample_median.png")
