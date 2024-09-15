import arviz
import matplotlib.pyplot as plt

azdata = arviz.from_netcdf("output/samples.nc")
arviz.plot_pair(azdata, kind="kde", divergences=False, marginals=True, 
                var_names=["fsed", "vr", "mmr_ch4", "factor", "sigma"])
plt.savefig("output/pairplot.png")
