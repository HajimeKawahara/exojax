import matplotlib.pyplot as plt
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol
from exojax.opacity import OpaPremodit
from exojax.plot.opaplot import plot_lbd
from jax import config

config.update("jax_enable_x64", True)


def save_lbd(
    filename,
    nu_grid,
    lbd_coeff,
    multi_index_uniqgrid,
    elower_grid,
    ngamma_ref_grid,
    n_Texp_grid,
):
    np.savez(
        filename,
        lbd_coeff=lbd_coeff,
        elower_grid=elower_grid,
        ngamma_ref_grid=ngamma_ref_grid,
        n_Texp_grid=n_Texp_grid,
        nu_grid=nu_grid,
        multi_index_uniqgrid=multi_index_uniqgrid,
    )


# CO case
nu_grid, wav, resolution = wavenumber_grid(
    22900.0, 27000.0, 200000, unit="AA", xsmode="premodit"
)
mdb = MdbExomol(".database/CO/12C-16O/Li2015/", nurange=nu_grid)
opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1000.0], dit_grid_resolution=0.2)
lbd, midx, gegrid, gngamma, gn, R, pm = opa.opainfo
plot_lbd(lbd, gegrid, gngamma, gn, midx, nu_grid)
plt.savefig("premodit_lbd_co.png", bbox_inches="tight", pad_inches=0.0)
plt.close()
filen = "premodit_lbd_coeff_co.npz"
save_lbd(filen, nu_grid, lbd, midx, gegrid, gngamma, gn)
# H2O case
nu_grid, wav, resolution = wavenumber_grid(
    14500.0, 15500.0, 100000, unit="AA", xsmode="premodit"
)
mdb = MdbExomol(".database/H2O/1H2-16O/POKAZATEL/", nurange=nu_grid)
opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1000.0], dit_grid_resolution=0.2)
lbd, midx, gegrid, gngamma, gn, R, pm = opa.opainfo
plot_lbd(lbd, gegrid, gngamma, gn, midx, nu_grid)
plt.savefig("premodit_lbd_h2o.png", bbox_inches="tight", pad_inches=0.0)
plt.close()
filen = "premodit_lbd_coeff_h2o.npz"
save_lbd(filen, nu_grid, lbd, midx, gegrid, gngamma, gn)
