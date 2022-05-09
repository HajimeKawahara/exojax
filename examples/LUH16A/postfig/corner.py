import numpy as np
import matplotlib.pyplot as plt
import arviz


p=np.load("~/fig/npz/savepos.npz",allow_pickle=True)["arr_0"][0]
#centered = az.load_arviz_data("centered_eight")

rc = {
    "plot.max_subplots": 250,
}
#arviz.style.use("arviz-darkgrid")
arviz.rcParams.update(rc)
axes=arviz.plot_pair(p,kind='kde',divergences=False,marginals=True,textsize=18)
fig = axes.ravel()[0].figure
fig.savefig("cornerall.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
