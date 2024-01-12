"""evaluating robust T range

"""
# %%
import numpy as np
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import worst_tilde_line_strength_first
from exojax.spec.lbderror import worst_tilde_line_strength_second
from exojax.spec.lbderror import evaluate_trange


N = 50
K = N
M = 120
L = 29
crit = 0.01
Trangel = 100.0
Trangelwt = Trangel + 0.1
Trangeh = 2000.0
Trangehwt = Trangeh + 0.1
Tsearchl = 100.0
Tsearchh = 5000.0
filename = "elower_grid_trange.npz"

def generate_elower_grid_trange(N, K, M, L, crit, Trangel, Trangelwt, Trangeh, Trangehwt, Tsearchl, Tsearchh, filename):
    Twtarr = np.logspace(np.log10(Trangelwt), np.log10(Trangehwt), N)
    Trefarr = np.logspace(np.log10(Trangel), np.log10(Trangeh), K)
    Tarr = np.logspace(np.log10(Tsearchl), np.log10(Tsearchh), M)
    dEarr = np.linspace(100, 1500, L)
    arr = np.zeros((2, N, K, L, 3))
    for idE, dE in enumerate(dEarr):
        print(idE, "/", L, dE)
        for iTref, Tref in enumerate(Trefarr):
            for iTwt, Twt in enumerate(Twtarr):
                x = single_tilde_line_strength_zeroth(1.0 / Tarr, 1.0 / Twt, 1.0 / Tref, dE)
                Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
                arr[0, iTwt, iTref, idE, 0] = Tl
                arr[1, iTwt, iTref, idE, 0] = Tu

                x = worst_tilde_line_strength_first(Tarr, Twt, Tref, dE * 2)
                Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
                arr[0, iTwt, iTref, idE, 1] = Tl
                arr[1, iTwt, iTref, idE, 1] = Tu

                x = worst_tilde_line_strength_second(Tarr, Twt, Tref, dE * 3)
                Tl, Tu = evaluate_trange(Tarr, x, crit, Twt)
                arr[0, iTwt, iTref, idE, 2] = Tl
                arr[1, iTwt, iTref, idE, 2] = Tu
            # print(Tl, Twt, Tu)

    np.savez(filename, arr, Tarr, Twtarr, Trefarr, dEarr)

generate_elower_grid_trange(N, K, M, L, crit, Trangel, Trangelwt, Trangeh, Trangehwt, Tsearchl, Tsearchh, filename)

# %%
