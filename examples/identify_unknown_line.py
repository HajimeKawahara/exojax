"""
Idnetifying unknown lines in Luhman16B spectrum residuals
==============================================================================================================================

This example tries to identify unknown absorption lines in the residuals of the Luhman16B spectrum after fitting known molecular lines.
It uses ExoMolHR database.

"""

# %%
from tabnanny import check
from jax import config

config.update("jax_enable_x64", True)

# sphinx_gallery_thumbnail_path = '_static/unknown_lines_luh16B.png'

# %%

import pandas as pd
import numpy as np

# %%
# loading model and target data
# ----------------------------------
dat = pd.read_csv(
    "luh16B_spectrum_model.dat",
    delimiter=r"\s+",
    names=("wav", "flux", "model"),
    header=0,
    engine="python",
)
wavd = dat["wav"].to_numpy()
model = dat["model"].to_numpy()
flux = dat["flux"].to_numpy()

# %%
# plotting
# -----------------------------------------------
# 
# By inspecting the residuals between the data and the model, we identfied unknown absorption lines, for instance, around 23058AA.
#

import matplotlib.pyplot as plt


def plotfig(filename, xrange=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(wavd, flux, ".", color="gray", label="Data", alpha=0.5)
    ax.plot(wavd, np.asarray(model), lw=1, label="Model", alpha=0.8, color="C1")
    ax.set_xlim(wavd.min(), wavd.max())
    ax.set_ylabel("Flux (normalized)")
    ax.legend(loc="best")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(
        wavd, flux - np.asarray(model), ".", color="gray", label="residual", alpha=0.5
    )
    if xrange is not None:
        ax.set_xlim(xrange[0], xrange[1])
        ax2.set_xlim(xrange[0], xrange[1])
    ax2.set_xlabel("Wavelength [angstrom]")
    ax2.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"saved to: {filename}")

checkrange = [23030, 23180]
plotfig("single_spectrum_sample_median_1.png", checkrange)

# %%
# uses ExoMolHR to check whether there are unknown lines in this wavelength range
# -------------------------------------------------------------------------------------

from exojax.database.exomolhr.api import list_exomolhr_molecules
from exojax.database.exomolhr.api import list_exomolhr_isotopes

molecules = list_exomolhr_molecules()
print(molecules)

iso_dict = list_exomolhr_isotopes(molecules)
print(iso_dict)


# %%
# Downloads all the available species
# -----------------------------------------------

from exojax.database import XdbExomolHR
from exojax.utils.grids import wavenumber_grid
import matplotlib.pyplot as plt

nus, _, _ = wavenumber_grid(wavd[0], wavd[-1], 10, xsmode="premodit", unit="AA")
temperature = 1300.0

k = 0
xdbs = {}
for molecule in iso_dict:
    isos = iso_dict[molecule]
    for j, iso in enumerate(isos):
        try:
            xdb = XdbExomolHR(iso, nus, temperature, crit=1.0e-24)
            xdbs[iso] = xdb
        except:
            k = k + 1
            print(f"No line? {iso}")
print(k, "molecules have no lines")

# %%
# plotting the result
# -----------------------------------------------

lslist = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
lwlist = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
markers_list = [".", "o", "s", "D", "^", "v", "<", ">"]

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(2, 1, 2)
ax.plot(
        wavd, flux - np.asarray(model), ".", color="gray", label="residual", alpha=0.5
    )
ax.set_xlim(checkrange[0], checkrange[1])

ax2 = fig.add_subplot(2, 1, 1)
for molecule in iso_dict:
    isos = iso_dict[molecule]
    for j, iso in enumerate(isos):
        try:
            xdb = xdbs[iso]
            ax2.plot(1.e8/xdb.nu_lines, xdb.line_strength, markers_list[j], label=iso, ls=lslist[j], lw=lwlist[j])
        except:
            print(f"No line? {iso}")
            continue

ax2.set_xlim(checkrange[0], checkrange[1])
ax2.set_yscale("log")
ax.set_xlabel("wavelength (AA)")
ax2.set_ylabel("Line Strength (cm/molecule)")
ax2.legend(loc="upper right", fontsize=8)
plt.savefig("unknown_lines_luh16B.png", dpi=200)
plt.close(fig)
print("saved to: unknown_lines_luh16B.png")

# We did not find any known molecular lines in ExoMolHR that correspond to the unknown absorption lines in the residuals.