import matplotlib.pyplot as plt
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from jax import config

config.update("jax_enable_x64", True)

nu_grid, wav, resolution = wavenumber_grid(
    14500.0, 15500.0, 100000, unit="AA", xsmode="premodit"
)
mdb = MdbExomol(".database/H2O/1H2-16O/POKAZATEL/", nurange=nu_grid)
opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1000.0], dit_grid_resolution=0.2)
(
    lbd_coeff,
    multi_index_uniqgrid,
    elower_grid,
    ngamma_ref_grid,
    n_Texp_grid,
    R,
    pmarray,
) = opa.opainfo

print(lbd_coeff.shape)

np.savez(
    "premodit_lbd_coeff.npz",
    lbd_coeff=lbd_coeff,
    elower_grid=elower_grid,
    ngamma_ref_grid=ngamma_ref_grid,
    n_Texp_grid=n_Texp_grid,
    nu_grid=nu_grid,
    multi_index_uniqgrid=multi_index_uniqgrid,
)

