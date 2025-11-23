# multi wavelength range
# Create the model data of absorption separating each lines
# 2024/08/29 created based on "CH4Gascellmodel_repeat_2407rev.py"
##opapremodit, gridboost

from exojax.utils.grids import wavenumber_grid
from exojax.database.hitemp.api import MdbHitemp
from exojax.database.core.line_strength import line_strength
from exojax.postproc.specop import SopInstProfile
from Trans_model_1Voigt_HITEMP_nu_2408rev_test import (
    Trans_model_MultiVoigt_test,
    create_mdbs_multi,
    calc_dnumber_isobaric,
)
import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import Tref_original
import matplotlib.collections as mc
from jax import config

config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Calculate the Line strength S(T)
def S_Tcalc(nu, S_0, T):
    logeS_0 = jnp.log(S_0)
    qr = mdb.qr_interp_lines(T, Tref_original)
    return line_strength(T, logeS_0, nu, mdb.elower, qr, Tref_original)


wmin = 1621.78
wspan = 0.15
adjustrange = 0.2
Resolution = 0.0025
nu_offset = 0.00
Twt = 1000
gridboost = 10

# model spectra setting
VMR = 1  # volume mixing ratio
kB = 1.380649e-16
L = 50 * 1  # cm
P0 = 0.423  # @296K
# Tarr = jnp.array([1000, 1000, 1000])
Tarr = jnp.array([720.5, 720.0, 727.2, 734.3, 733.8, 731.4, 729.7, 725.2]) + 273.15
# Tarr =jnp.array([320,383,414,427,427,414,385,324]) +273#CH3-9 of  162178-93 at 2023/07/24/15:21:02
# Tarr =jnp.array([581, 664, 702, 721, 720, 705, 668, 590]) +273#CH3-9 of  162141-51 at 2023/07/25/17:02:00

roundorder = 5  # digit to round up
Nx = round(((wspan + 2 * adjustrange) / Resolution + 1))  # it should be even number
start_idx = round(adjustrange / Resolution)  # start index in wav array for triminng
slicesize = round((wspan / Resolution + 1))

Nx_boost = Nx * gridboost
ngas, P_total = calc_dnumber_isobaric(Tarr, P0, Tref_original)
P_self = P_total * VMR
nMolecule = ngas * VMR

wmax = round(wmin + wspan, 5)
wav_n = round(wspan / Resolution + 1)  # equal to the specified amount of data point
wav_adjust = jnp.linspace(wmin - adjustrange, wmax + adjustrange, Nx)
nu_adjust = 1e7 / wav_adjust[::-1]

nu_min = 1e7 / wmax
nu_max = 1e7 / wmin

# generate the wavenumber&wavelength grid for cross-section
nu_grid, wav, res = wavenumber_grid(
    nu_min - adjustrange,
    nu_max + adjustrange,
    # jnp.max(wavd),
    Nx_boost,
    unit="cm-1",
    xsmode="premodit",
    wavelength_order="ascending",
)

sop_inst = SopInstProfile(nu_grid)

# Read the line database
mdb = MdbHitemp(
    ".database/CH4/",
    nurange=nu_grid,
    gpu_transfer=False,  # Trueだと計算速度低下
)  # for obtaining the error of each line

# line amount for Voigt fitting
linenum = 1
# Calculate the line index in the order of the line strength at T=twt
S_T = S_Tcalc(jnp.exp(mdb.nu_lines), mdb.line_strength_ref_original, Twt)

strline_ind_array = jnp.argsort(S_T)[::-1][:linenum]
strline_ind_array_nu = jnp.sort(strline_ind_array)

mdb_weak, nu_center_voigt, mdb_voigt = create_mdbs_multi(mdb, strline_ind_array_nu)
# opa = create_opapremodit(mdb_weak, nu_grid, Tarr)
from exojax.opacity import OpaPremodit

opa = OpaPremodit(
    mdb=mdb_voigt,
    nu_grid=nu_grid,
    diffmode=0,  # i-th Taylor expansion is used for the weight, default is 0.
    auto_trange=(np.min(Tarr), np.max(Tarr)),
    # manual_params=(160, Tref_original, np.max(Tarr)),
)  # opacity calculation
print(opa.Tref, opa.Tref_broadening)
#import sys
#sys.exit()
#opa.Tref = Tref_original
#opa.Tref_broadening = Tref_original


# Fixed values definition
nspec = 1  # Number of spectra is 1, so loops over nspec can be removed
valrange = 1.0  # Value range, adjust as needed
nu_span = 0.5  # Span of nu, adjust as needed

# Parameter definitions with fixed values
offrange = 0.1

alphas = jnp.ones((linenum))  # Fixed value as an array of ones
gamma_selfs = mdb_voigt.gamma_self
ns = mdb_voigt.n_air
# Polynomial coefficients for nspec = 1
coeffs = [
    {
        "a": 0.0,  # Fixed value
        "b": 0.0,  # Fixed value
        "c": 0.0,  # Fixed value
        "d": 1.0,  # Fixed value
    }
]

wavmodel = wav_adjust[start_idx : start_idx + slicesize]
nu_model = 1e7 / wavmodel


xsmatrix_opa, xsmatirx_lpf = Trans_model_MultiVoigt_test(
    nu_offset,
    alphas,
    mdb_voigt.gamma_air,
    gamma_selfs,
    ns,
    Tarr,
    P_total,
    P_self,
    L,
    nMolecule,
    nu_grid,
    nu_model,
    mdb_voigt,
    opa,
    sop_inst,
)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
# plot the measured spectra

ax.plot(
    wavmodel,
    xsmatrix_opa[::-1],
    "-",
    alpha=1.0,
    linewidth=2,
    color="C0",
    label="opapremodit cross-section",
)


ax.plot(
    wavmodel,
    xsmatirx_lpf[::-1],
    "-",
    alpha=1.0,
    linewidth=2,
    ls="dashed",
    color="C1",
    label="lpf cross-section",
)


# Plot range
ymin = 0.0
ymax = 1.01
yspan = ymax - ymin
# plt.ylim(ymin, ymax)
plt.xlim(wmin, wmax)
# plt.xlim(wmin - adjustrange, wmax + adjustrange)


# plot the line centers
wavlinecenter1 = 1 / mdb.nu_lines * 1.0e7  # convert to wavelength (descending)
wavlines1 = [
    [(wavlinecenter1[i], ymax - yspan * 0.05), (wavlinecenter1[i], ymax)]
    for i in range(linenum)
]
lc1 = mc.LineCollection(
    wavlines1, colors="gray", linewidths=1, linestyle="-", label="line centers"
)
ax.add_collection(lc1)

# plot setting
plt.grid(which="major", axis="both", alpha=0.7, linestyle="--", linewidth=1)
ax.grid(which="minor", axis="both", alpha=0.3, linestyle="--", linewidth=1)
ax.minorticks_on()
plt.xlabel("wavelength (nm)", fontsize=16)
plt.ylabel("Transmittance", fontsize=16)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=8, ncol=4)
ax.get_xaxis().get_major_formatter().set_useOffset(
    False
)  # To avoid exponential labeling
plt.tick_params(labelsize=16)
plt.text(
    0.99,
    1.01,
    "Line N=" + str(len(mdb.nu_lines)),
    fontsize=10,
    va="bottom",
    ha="right",
    transform=ax.transAxes,
)

# Read the Wavelength range as str
wavmin_str = str(wmin).replace(".", "")
wavmax_str = str(wmax).replace(".", "")

direc = "Results/TransSpectra/"
Fname = (
    "TEST_"
    + "CH4-VMR1_L="
    + str(L)
    + "cm_T1000K_P1430_"
    + wavmin_str
    + "-"
    + wavmax_str
    + "_00025_hitemp_gamma-air_linenum"
    + str(linenum)
    + "_VSmeasure_adjustrange_opa-Tbroad-Tref-Treforig"
)

# save the Graph
plt.savefig(
    direc + Fname + "_eachline_wong2019_2408rev_VScrosssection.jpg", bbox_inches="tight"
)
print("Saved .jpg as ", Fname, "_eachline.jpg")

# plt.show()
plt.clf()  # clear the plot
plt.close()  # close the plot window

"""
#Save the plot data as txt file
Modeldirectry = "Results/TransSpectra/"
f = open(Modeldirectry + Fname + ".txt",'w')
for i in range(wavmodel.size): 
    f.write(str(wavmodel[i]) + "," +str(transmodel_all[i]) +  "\n")
    #f.write(str(wavmodel[i]) + "," +str(noisemodel[i]) +  "\n")
f.close()
print("Saved .txt as ", Fname)
"""
