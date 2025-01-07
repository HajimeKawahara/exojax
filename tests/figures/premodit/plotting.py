import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter


def log_formatter(value, tick_number):
    return f"{10**value:.1f}"


dat = np.load("premodit_lbd_coeff.npz")

lbd_coeff = dat["lbd_coeff"]
egrid = dat["elower_grid"]
ngrid = dat["ngamma_ref_grid"]
nTexp = dat["n_Texp_grid"]
multi_index_uniqgrid = dat["multi_index_uniqgrid"]
nu_grid = dat["nu_grid"]

lbd_coeff[lbd_coeff == -np.inf] = np.nan
lbd = np.exp(lbd_coeff[0, :, :, :])
# integrates over ngamma_ref
arr = np.nansum(lbd, axis=1)
arr = np.log10(arr)
# integrate over Elower
arrx = np.nansum(lbd, axis=0)
arrx = np.log10(arrx)

number_of_ticks = 10
n = int(len(nu_grid) / number_of_ticks)
log_ticks = np.log10(nu_grid[::n])

fig = plt.figure(figsize=(15, 3))
gs = gridspec.GridSpec(1, 5, figure=fig)
ax = fig.add_subplot(gs[0, :4])
ax.set_xticks(log_ticks)

# Warning: interpolation = "none" in imshow is very important, otherwise the fine structure is washed out.
c = ax.imshow(
    arr.T,
    aspect="auto",
    cmap="inferno",
    interpolation="none",
    extent=[np.log10(nu_grid[0]), np.log10(nu_grid[-1]), egrid[-1], egrid[0]],
    vmin=-70,
    vmax=-20,
)
cbar = plt.colorbar(c)
cbar.set_label("log10(LBD (cm/bin))")
ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
ax.set_xlabel("wavenumber (cm-1)")
ax.set_ylabel("elower (cm-1)")
plt.gca().invert_yaxis()

ax = fig.add_subplot(gs[0, 4])
c = ax.imshow(
    arrx.T,
    aspect="auto",
    cmap="inferno",
    interpolation="none",
    extent=[0, len(multi_index_uniqgrid) - 1, egrid[-1], egrid[0]],
    vmin=-70,
    vmax=-20,
)
ax.xaxis.set_ticklabels([])
ax.axes.get_xaxis().set_ticks([])
ax.set_xlabel("index for width", labelpad=12)
ax2 = ax.twiny()
ax2.axes.get_xaxis().set_ticks([])
ax2.set_xlabel("index for power", labelpad=12)

cbar = plt.colorbar(c)
cbar.set_label("log10(LBD (cm/bin))")
for i, miu in enumerate(multi_index_uniqgrid):
    iwidth = miu[0]
    ipower = miu[1]
    ax.text(i, egrid[0], str(iwidth), ha="center", va="top")
    ax.text(i, egrid[-1], str(ipower), ha="center", va="bottom")
ax.set_ylabel("elower (cm-1)")
plt.gca().invert_yaxis()

plt.savefig("premodit_lbd_coeff.png", bbox_inches="tight", pad_inches=0.0)
