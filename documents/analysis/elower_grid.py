"""evaluating robust T range
"""
#%%
import numpy as np
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import worst_tilde_line_strength_first
from exojax.spec.lbderror import worst_tilde_line_strength_second
from exojax.spec.lbderror import evaluate_trange

N = 50
Twtarr = np.logspace(np.log10(100.1), np.log10(2000.1), N)
K = N
Trefarr = np.logspace(np.log10(100.), np.log10(2000.), K)

#K = 3
#Trefarr = np.array([400.0,800.0,1200.0])

M = 120
Tarr = np.logspace(np.log10(100.), np.log10(5000.), M)
L = 29
dEarr = np.linspace(100, 1500, L)

#L = 1
#dEarr = np.array([1500.0])


arr = np.zeros((2, N, K, L, 3))
crit = 0.01
for idE, dE in enumerate(dEarr):
    print(idE, "/", L, dE)
    for iTref, Tref in enumerate(Trefarr):
        for iTwt, Twt in enumerate(Twtarr):
            x = single_tilde_line_strength_zeroth(1. / Tarr, 1.0 / Twt,
                                                  1.0 / Tref, dE)
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0, iTwt, iTref, idE, 0] = Tl
            arr[1, iTwt, iTref, idE, 0] = Tu

            x = worst_tilde_line_strength_first(Tarr, Twt, Tref, dE*2)            
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0, iTwt, iTref, idE, 1] = Tl
            arr[1, iTwt, iTref, idE, 1] = Tu

            x = worst_tilde_line_strength_second(Tarr, Twt, Tref, dE*3)
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0, iTwt, iTref, idE, 2] = Tl
            arr[1, iTwt, iTref, idE, 2] = Tu
            #print(Tl, Twt, Tu)

np.savez("elower_grid_trange.npz", arr, Tarr, Twtarr, Trefarr, dEarr)

# %%
