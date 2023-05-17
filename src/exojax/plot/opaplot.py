import numpy as np


def plot_broadening_parameters_grids(ngamma_ref_grid, n_Texp_grid, nu_grid,
                                     resolution, gamma_ref_in, n_Texp_in, crit,
                                     figname):
    if 2 * len(gamma_ref_in) > crit:
        n = int(len(gamma_ref_in) / crit)
        gamma_ref = gamma_ref_in[::n]
        n_Texp = n_Texp_in[::n]
        print("Show broadening parameter sampling points, n_sample=", str(n))
        print(len(gamma_ref_in), "->", len(gamma_ref), "pts")
    else:
        gamma_ref = gamma_ref_in
        n_Texp = n_Texp_in

    maxgamma = ngamma_ref_grid / resolution * np.max(nu_grid)
    gammag_max, n_Texpg_max = _mesh_grid(n_Texp_grid, maxgamma)

    mingamma = ngamma_ref_grid / resolution * np.min(nu_grid)
    gammag_min, n_Texpg_min = _mesh_grid(n_Texp_grid, mingamma)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(gamma_ref, n_Texp, ".", alpha=0.3, label="data")
    plt.plot(gammag_min, n_Texpg_min, "x", label="grid (min)", color="gray")
    plt.plot(gammag_max, n_Texpg_max, "+", label="grid (max)", color="black")

    plt.xscale("log")
    plt.xlabel("$\gamma_\mathrm{ref}$  $(\mathrm{cm}^{-1})$")
    plt.ylabel("temperature exponent")
    plt.legend()
    plt.savefig(figname)
    plt.show()


def _mesh_grid(n_Texp_grid, gamma):
    X, Y = np.meshgrid(gamma, n_Texp_grid)
    grid = np.column_stack([X.flatten(), Y.flatten()])
    gammag = grid[:, 0]
    n_Texpg = grid[:, 1]
    return gammag, n_Texpg
