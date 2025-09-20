from exojax.opacity import OpaPremodit
from exojax.rt import ArtEmisPure
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol
from exojax.utils.astrofunc import gravity_jupiter
import jax.profiler

from jax import config

config.update("jax_enable_x64", True)
#from jax_smi import initialise_tracking
#initialise_tracking()

Nnus = 100000
nu_grid, wav, resolution = wavenumber_grid(
#    1900.0, 2300.0, Nnus, unit="cm-1", xsmode="premodit"
    2050.0, 2150.0, Nnus, unit="cm-1", xsmode="premodit"
)
mdb_co = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)
opa_co = OpaPremodit(
    mdb_co,
    nu_grid,
    auto_trange=[500.0, 1500.0],
    dit_grid_resolution=1.0,
)

gravity = gravity_jupiter(1.0, 10.0)
art = ArtEmisPure(
    nu_grid=nu_grid, pressure_top=1.0e-5, pressure_btm=1.0e1, nlayer=200, nstream=8
)

import time
import jax.numpy as jnp
import tqdm

import matplotlib.pyplot as plt

Ntry = 100
T = jnp.array(range(0, Ntry)) + 1100.0
ts = time.time()
if True:
    fluxarr = []
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in tqdm.tqdm(range(Ntry)):
        temperature = art.powerlaw_temperature(T[i], 0.1)
        mixing_ratio = art.constant_profile(0.01)
        xsmatrix = opa_co.xsmatrix(temperature, art.pressure)
        dtau = art.opacity_profile_xs(xsmatrix, mixing_ratio, mdb_co.molmass, gravity)
        flux = art.run(dtau, temperature)
        fluxarr.append(flux)
        flux.block_until_ready()
  
#    plt.plot(nu_grid,flux)


te = time.time()
print("time=", te - ts)
jax.profiler.save_device_memory_profile("memory_opa_and_art.prof")

plot = True
if plot:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for i in range(Ntry):
        plt.plot(nu_grid, fluxarr[i])
    #plt.yscale("log")
    plt.ylim(0,30000.0)
    plt.xlim(2080,2090)
    plt.savefig("forward_opa_ard_art_200000.png")
