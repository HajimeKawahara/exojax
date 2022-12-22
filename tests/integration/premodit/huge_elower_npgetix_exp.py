""" This test accounts for Issue #288, bug fix large elower value using f32, 
    The bug was due to the overflow in the function when computing,
"""

from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from exojax.spec.rtransfer import pressure_layer
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec import api
from exojax.spec import initspec, molinfo, premodit
from exojax.spec import molinfo
from exojax.utils.astrofunc import getjov_gravity
from exojax.utils.constants import RJ, pc

wls, wll, Ndata = 15035, 15040, 100
wavd = np.linspace(wls, wll, Ndata)
nflux = np.random.rand(Ndata)
nusd = jnp.array(1.e8 / wavd[::-1])

NP = 100
Parr, dParr, k = pressure_layer(NP=NP)

Nx = 2000
nus, wav, reso = wavenumber_grid(np.min(wavd) - 5.0,
                                 np.max(wavd) + 5.0,
                                 Nx,
                                 unit="AA",
                                 xsmode="modit")

Rinst = 100000.
beta_inst = resolution_to_gaussian_std(Rinst)

#Reference pressure for a T-P model
Pref = 1.0  #bar
ONEARR = np.ones_like(Parr)

#Load H2O data with premodit
molmassH2O = molinfo.molmass_major_isotope("H2O")
mdbH2O_orig = api.MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nus, gpu_transfer=False)
print('N_H2O=', len(mdbH2O_orig.nu_lines))
#
Tgue = 3000.
interval_contrast = 0.1
dit_grid_resolution = 0.1
lbd_H2O, multi_index_uniqgrid_H2O, elower_grid_H2O, \
ngamma_ref_grid_H2O, n_Texp_grid_H2O, R_H2O, pmarray_H2O = initspec.init_premodit(
    mdbH2O_orig.nu_lines,
    nus,
    mdbH2O_orig.elower,
    mdbH2O_orig.alpha_ref,
    mdbH2O_orig.n_Texp,
    mdbH2O_orig.Sij0,
    Ttyp=Tgue,
    interval_contrast=interval_contrast,
    dit_grid_resolution=dit_grid_resolution,
    warning=False)

from jax import random
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from exojax.spec.rtransfer import dtauM_mmwl, rtrun
from exojax.spec import planck, response
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
from jax import vmap

#response settings
vsini_max = 100.0
vr_array = velocity_grid(reso, vsini_max)


#core driver frun
def frun_lbl(VMR_H2O):
    Tarr = 3000 * (Parr / Pref)**0.1
    RV = 0.0
    vsini = 2.0
    mmw = 2.33
    Mp = 0.155 * 1.99e33 / 1.90e30
    Rp = 0.186 * 6.96e10 / 6.99e9
    u1 = 0.0
    u2 = 0.0
    ga = getjov_gravity(Rp, Mp)

    #atmosphere (T-P profile)
    VMR_H = 0.16
    VMR_H2 = 0.84
    PH = Parr * VMR_H
    PHe = Parr * (1 - VMR_H - VMR_H2)
    PHH = Parr * VMR_H2
    mmw = mmw * ONEARR

    #H2O with premodit
    qtarr = vmap(mdbH2O_orig.qr_interp)(Tarr)
    xsm_H2O = premodit.xsmatrix(Tarr, Parr, R_H2O, pmarray_H2O, lbd_H2O, nus,
                                ngamma_ref_grid_H2O, n_Texp_grid_H2O,
                                multi_index_uniqgrid_H2O, elower_grid_H2O,
                                molmassH2O, qtarr)
    dtau = dtauM_mmwl(dParr, jnp.abs(xsm_H2O), VMR_H2O * ONEARR, mmw, ga)

    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)
    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
    mu = response.ipgauss_sampling(nusd, nus, Frot, beta_inst, RV)
    adjust_continuum1 = 1.
    adjust_continuum2 = 1.
    a = (adjust_continuum1 - adjust_continuum2) / (wavd[0] - wavd[-1])
    b = adjust_continuum1 - a * wavd[0]
    f_adjust_continuum = lambda x: a * x + b
    mu = mu / jnp.nanmax(mu) * f_adjust_continuum(wavd[::-1])

    return (mu)


def model_c(y1):
    VMR_H2O = 10**(numpyro.sample('VMR_H2O', dist.Uniform(-7., -1.)))
    mu = frun_lbl(VMR_H2O)

    sigma = numpyro.sample('sigma', dist.Exponential(1.0))
    numpyro.sample("y1", dist.Normal(mu, sigma), obs=y1)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 10, 20
kernel = NUTS(model_c, forward_mode_differentiation=True, max_tree_depth=7)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

mcmc.run(rng_key_, y1=nflux)
mcmc.print_summary()
