#%%
import numpy as np

dat = np.load("elower_grid_trange.npz")
arr = dat["arr_0"]
Tarr = dat["arr_1"]
Twtarr = dat["arr_2"]
Trefarr = dat["arr_3"]
dEarr = dat["arr_4"]

#arr[0:2, iTwt, iTref, idE, 0] = Tl, Tu
print(dEarr)
print(np.shape(arr))
# %%
import matplotlib.pyplot as plt


def optimal_params(Tl, Tu, diffmode=2):
    print(np.shape(arr))
    maskl = (arr[0, :, :, :, diffmode] <= Tl_in)
    masku = (arr[1, :, :, :, diffmode] >= Tu_in)
    mask = maskl * masku

    for i in range(len(dEarr)):
        k = -i - 1
        j = np.sum(mask[:, :, k])
        Tlarr = arr[0, :, :, k, diffmode]
        Tuarr = arr[1, :, :, k, diffmode]

        if i==0:
            #:,: = Y, X = Twt, Tref
            extent=[Trefarr[0],Trefarr[-1],Twtarr[-1],Twtarr[0]]
            fig = plt.figure(figsize=(15,5))
            ax = fig.add_subplot(131)
            c = plt.imshow(Tlarr, cmap="rainbow", extent=extent)
            pltadd(c)
            ax = fig.add_subplot(132)
            c = plt.imshow(Tuarr, cmap="rainbow", extent=extent)
            pltadd(c)
            ax = fig.add_subplot(133)
            c = plt.imshow(Tuarr-Tlarr, cmap="rainbow", extent=extent)
            pltadd(c)
            
            plt.show()
            #print(Tlarr)

        #print(dEarr[k], j)
        Tla = Tlarr[mask[:, :, k]]
        Tua = Tuarr[mask[:, :, k]]
        Tarr = np.array([Tla, Tua]).T
        print(Tarr, dEarr[k])

    arrx = arr[0, :, :, :, diffmode]
    
def pltadd(c):
    plt.colorbar(c,shrink=0.7)
    plt.ylabel("Twt (K)")
    plt.xlabel("Tref (K)")
    plt.gca().invert_yaxis()


Tl_in = 500.0  #K
Tu_in = 1000.0  #K
diffmode = 2
optimal_params(Tl_in, Tu_in, diffmode)

# %%

# %%
