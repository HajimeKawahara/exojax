"""
Example that OpaPremoditStitch can compute cross section matrix, but ends up OoM when using PreModit
tested using A100 40GB
"""


from exojax.opacity import OpaPremodit
from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol
import jax.numpy as jnp

from jax_smi import initialise_tracking
initialise_tracking()

from jax import config 
config.update("jax_enable_x64", True)

N=30000
nus, wav, res = wavenumber_grid(22000.0, 22500.0, N, unit="AA", xsmode="premodit")
mdb = MdbExomol(".database/H2O/1H2-16O/POKAZATEL", nus)

nlayer = 10000
Tarr = jnp.ones(nlayer)*1000.0
Parr = jnp.ones(nlayer)
print(Tarr.shape)

# OoM
#opa = OpaPremodit(mdb, nus, auto_trange=[500,1300], alias="close", dit_grid_resolution=1.0)
#xsm = opa.xsmatrix(Tarr, Parr)

# can compute
ndiv=20
opas = OpaPremodit(mdb, nus, nstitch=ndiv, auto_trange=[500,1300], cutwing = 0.05)
#Tarr = jnp.array([1000.0, 1100.0])
#Parr = jnp.array([1.0, 1.5])
xsm_s = opas.xsmatrix(Tarr, Parr)
