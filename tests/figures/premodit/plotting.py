import matplotlib.pyplot as plt
import numpy as np
from exojax.plot.opaplot import plot_lbd

dat = np.load("premodit_lbd_coeff.npz")

lbd_coeff = dat["lbd_coeff"]
elower_grid = dat["elower_grid"]
ngamma_ref_grid = dat["ngamma_ref_grid"]
n_Texp_grid = dat["n_Texp_grid"]
multi_index_uniqgrid = dat["multi_index_uniqgrid"]
nu_grid = dat["nu_grid"]


plot_lbd(lbd_coeff, elower_grid, ngamma_ref_grid, n_Texp_grid, multi_index_uniqgrid, nu_grid)
plt.savefig("premodit_lbd_coeff.png", bbox_inches="tight", pad_inches=0.0)