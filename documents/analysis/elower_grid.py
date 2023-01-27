"""evaluating robust T range
"""
#%%
import numpy as np
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import worst_tilde_line_strength_first
from exojax.spec.lbderror import worst_tilde_line_strength_second
from exojax.spec.lbderror import evaluate_trange

N = 30
Twtarr = np.logspace(np.log10(100.1), np.log10(2000.1), N)
Trefarr = np.logspace(np.log10(100.1), np.log10(2000.1), N)
M = 100
Tarr = np.logspace(np.log10(100.), np.log10(3000.), M)
L = 15
dEarr = np.linspace(100, 1500, L)
arr = np.zeros((2, N, N, L, 3))
crit = 0.01

for idE, dE in enumerate(dEarr):
    print(idE, "/", L)
    for iTref, Tref in enumerate(Trefarr):
        for iTwt, Twt in enumerate(Twtarr):
            x = single_tilde_line_strength_zeroth(1. / Tarr, 1.0 / Twt,
                                                  1.0 / Tref, dE)
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0:2, iTwt, iTref, idE, 0] = Tl, Tu
            x = worst_tilde_line_strength_first(Tarr, Twt, Tref, dE*2)            
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0:2, iTwt, iTref, idE, 1] = Tl, Tu
            x = worst_tilde_line_strength_second(Tarr, Twt, Tref, dE*3)
            Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
            arr[0:2, iTwt, iTref, idE, 2] = Tl, Tu

np.savez("elower_grid_trange.npz", arr, Tarr, Twtarr, Trefarr, dEarr)

# %%
