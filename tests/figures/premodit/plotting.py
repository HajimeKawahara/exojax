import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

dat=np.load("premodit_lbd_coeff.npz")

lbd_coeff = dat["lbd_coeff"]
egrid = dat["elower_grid"]
ngrid = dat["ngamma_ref_grid"]
nTexp = dat["n_Texp_grid"]
multi_index_uniqgrid = dat["multi_index_uniqgrid"]
nu_grid = dat["nu_grid"]

#print(multi_index_uniqgrid) #contains [width, n_Texp]
lbd_coeff[lbd_coeff==-np.inf] = np.nan
print(np.nanmin(lbd_coeff), np.nanmax(lbd_coeff))
lbd =np.exp(lbd_coeff[0,:,:,:])

# integrates over ngamma_ref
arr = np.nansum(lbd,axis=1)
#arr[arr == 0.0 ] = np.nan
arr = np.log10(arr)

fig = plt.figure(figsize=(15,3))
gs = gridspec.GridSpec(1, 5, figure=fig)



ax = fig.add_subplot(gs[0, :4])
# Warning: interpolation = "none" in imshow is very important, otherwise the fine structure is washed out.
c = ax.imshow(arr.T, aspect="auto", cmap="inferno", interpolation="none", extent=[nu_grid[0], nu_grid[-1], egrid[-1], egrid[0]], vmin=-70, vmax=-20)
cbar = plt.colorbar(c)
cbar.set_label("log10(LBD (cm/bin))")
plt.xlabel("wavenumber (cm-1)")
plt.ylabel("elower (cm-1)")

plt.gca().invert_yaxis()

arrx = np.nansum(lbd,axis=0)
arrx = np.log10(arrx)

ax = fig.add_subplot(gs[0, 4])
c = ax.imshow(arrx.T, aspect="auto", cmap="inferno", interpolation = "none", extent=[0, 15, egrid[-1], egrid[0]], vmin=-70, vmax=-20)
cbar = plt.colorbar(c)
cbar.set_label("log10(LBD (cm/bin))")
plt.ylabel("elower (cm-1)")
plt.gca().invert_yaxis()

plt.savefig("premodit_lbd_coeff.png", bbox_inches="tight", pad_inches=0.0)
