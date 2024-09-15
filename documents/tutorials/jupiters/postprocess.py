import arviz
import matplotlib.pyplot as plt
import plotjupiter
import arviz
from numpyro.diagnostics import hpdi
import jax.numpy as jnp
import numpy as np
from exojax.spec.unitconvert import nu2wav

dat = np.loadtxt("jupiter_corrected.dat")  # made by Jupiter_Hires_Modeling_NIR.ipynb
unmask_nus_obs = dat[:, 0]
unmask_spectra = dat[:, 1]
mask = (unmask_nus_obs < 6163.5) + ((unmask_nus_obs > 6166) & (unmask_nus_obs < 6174))
nus_obs = unmask_nus_obs[mask]
wav_obs = nu2wav(nus_obs, unit="AA")
spectra = unmask_spectra[mask]


azdata = arviz.from_netcdf("output/samples.nc")
arviz.plot_pair(azdata, kind="kde", divergences=False, marginals=True, 
                var_names=["fsed", "vr", "mmr_ch4", "factor", "sigma"])
plt.savefig("output/pairplot.png")

predictions = np.load("output/predictions.npz", allow_pickle=True)["arr_0"]
predictions = predictions.item()
median_mu1 = jnp.median(predictions["y1"], axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.95)

# prediction plot
plt = plotjupiter.plot_prediction(wav_obs, spectra, median_mu1, hpdi_mu1)

#unmask_wav_obs = nu2wav(unmask_nus_obs, unit="AA")
#plt.plot(unmask_wav_obs, unmask_spectra, ".", alpha=0.3, color="gray")
plt.savefig("output/Jupiter_fit_wav.png", bbox_inches="tight", pad_inches=0.1)
