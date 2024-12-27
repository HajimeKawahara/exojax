from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opart import OpartReflectPure
from exojax.spec.layeropacity import single_layer_optical_depth
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.utils.astrofunc import gravity_jupiter
import time
import tqdm
import jax.profiler
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax_smi import initialise_tracking
initialise_tracking()

class OpaLayer:
    # user defined class, needs to define self.nugrid
    def __init__(self, Nnus=100000):
        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            1900.0, 2300.0, Nnus, unit="cm-1", xsmode="premodit"
            #2050.0, 2150.0, Nnus, unit="cm-1", xsmode="premodit"

        )
        self.mdb_co = MdbExomol(".database/CO/12C-16O/Li2015", nurange=self.nu_grid)
        self.opa_co = OpaPremodit(
            self.mdb_co,
            self.nu_grid,
            auto_trange=[500.0, 1500.0],
            dit_grid_resolution=1.0,
            allow_32bit=True
        )
        self.gravity = gravity_jupiter(1.0, 10.0)


    #@partial(jit, static_argnums=(0,)) # this is not necessary and makes it significantly slow
    def __call__(self, params):
        temperature, pressure, dP, mixing_ratio = params
        xsv_co = self.opa_co.xsvector(temperature, pressure)
        dtau_co = single_layer_optical_depth(
            xsv_co, dP, mixing_ratio, self.mdb_co.molmass, self.gravity
        )
        return dtau_co

opalayer = OpaLayer(Nnus=100000)
opart = OpartReflectPure(opalayer, pressure_top=1.0e-5, pressure_btm=1.0e1, nlayer=200, nstream=8)
opart.change_temperature_range(400.0, 1500.0)
def layer_update_function(carry_tauflux, params):
    carry_tauflux = opart.update_layer(carry_tauflux, params)
    return carry_tauflux, None


Ntry = 100
T = jnp.array(range(0, Ntry)) + 1100.0
#mmr = jnp.ones(Ntry) * 0.01 + jnp.array(range(0, Ntry)) * 1.0e-5



ts = time.time()
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
if True:
    fluxarr = []
    for i in tqdm.tqdm(range(Ntry)):
        temperature = opart.clip_temperature(opart.powerlaw_temperature(T[i], 0.1))

        mixing_ratio = opart.constant_mmr_profile(0.01)
        layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
        flux = opart(layer_params, layer_update_function)
        fluxarr.append(flux)
        #flux.block_until_ready()
  
te = time.time()
print("time=", te - ts)
plot = True
if plot:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for i in range(Ntry):
        plt.plot(opalayer.nu_grid, fluxarr[i])
    #plt.yscale("log")
    plt.ylim(0,30000.0)
    plt.savefig("forward_opart.png")


jax.profiler.save_device_memory_profile("memory_opart.prof")
