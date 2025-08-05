"""this code repoduces the bug reported in #275. (solved)

"""

from jax import config
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol 
from exojax.opacity import initspec


config.update("jax_enable_x64", False)  # if True, no error.
crit = 1.0e-25  # If you increase the crit (such as 1.e-24), no error.

Tgue = 3000.0
wls, wll = 15020, 15050
Nx = 2000
nus, wav, reso = wavenumber_grid(wls, wll, Nx, unit="AA", xsmode="premodit")

mdbH2O_orig =MdbExomol(
    ".database/H2O/1H2-16O/POKAZATEL", nus, crit=crit, Ttyp=Tgue
)
print("N=", len(mdbH2O_orig.nu_lines))

interval_contrast = 0.1
dit_grid_resolution = 0.1

(
    lbd_H2O,
    multi_index_uniqgrid_H2O,
    elower_grid_H2O,
    ngamma_ref_grid_H2O,
    n_Texp_grid_H2O,
    R_H2O,
    pmarray_H2O,
) = initspec.init_premodit(
    mdbH2O_orig.nu_lines,
    nus,
    mdbH2O_orig.elower,
    mdbH2O_orig.alpha_ref,
    mdbH2O_orig.n_Texp,
    mdbH2O_orig.line_strength,
    Twt=Tgue,
    Tref=1000.0,
    Tref_broadening=1000.0,
    dit_grid_resolution=dit_grid_resolution,
    warning=False,
)
