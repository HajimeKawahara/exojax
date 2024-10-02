from cgitb import text
import arviz
import matplotlib.pyplot as plt
import plotjupiter
from loaddata import load_jupiter_reflection
import arviz
from numpyro.diagnostics import hpdi
import jax.numpy as jnp
import numpy as np

wav_obs, nus_obs, spectra, unmask_wav_obs, unmask_nus_obs, unmask_spectra, mask = (
    load_jupiter_reflection()
)



# PLOT: corner plot
azdata = arviz.from_netcdf("output/samples.nc")
var = ["fsed", "vr", "mmr_ch4", "factor", "sigma"]
label = ["$f_\mathrm{sed}$", "$v_r$", "MMR (CH4)", "$A$", "$\sigma$"]
ax = arviz.plot_pair(
    azdata, kind="kde", divergences=False, marginals=False, var_names=var, textsize=30
)
# label change
for i, ax_ in enumerate(ax.flatten()):
    if ax_.get_xlabel():
        ax_.set_xlabel(label[i - 12])
        # ax_.set_xlabel(label[i - 20]) #when marginals=True
    if ax_.get_ylabel():
        ax_.set_ylabel(label[int(i / 4) + 1])
        # ax_.set_ylabel(label[int(i / 5)]) #when marginals=True
plt.tick_params(labelsize=30)
plt.savefig("output/pairplot.png", bbox_inches="tight", pad_inches=0.1)

# PLOT: posterior plot
predictions = np.load("output/predictions.npz", allow_pickle=True)["arr_0"]
predictions = predictions.item()
median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.95)
plt = plotjupiter.plot_prediction(wav_obs, spectra, median_mu1, hpdi_mu1)

# unmask_wav_obs = nu2wav(unmask_nus_obs, unit="AA")
#plt.plot(unmask_wav_obs, unmask_spectra, ".", alpha=0.3, color="gray")
plt.savefig("output/Jupiter_fit_wav.png", bbox_inches="tight", pad_inches=0.1)

from exojax.utils.zsol import nsol
from exojax.utils.zsol import mass_fraction

n = nsol("AG89")
Xc = mass_fraction("C", n)
mmr_sol = Xc * 16 / 12
print("MMR_CH4 for 3 Zsol (AG89) = ", 3 * mmr_sol)
MMR5 = 10 ** (0.04)
MMR95 = 10 ** (0.22)
print("MMR (5-95%):", MMR5, "-", MMR95)
print("Z:", MMR5 * 0.01 / mmr_sol, "-", MMR95 * 0.01 / mmr_sol)
