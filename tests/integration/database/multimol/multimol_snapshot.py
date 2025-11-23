from jax import config

config.update("jax_enable_x64", True)

from exojax.utils.grids import wavenumber_grid
from exojax.database.multimol import MultiMol


mols = [["CO"]]
db = [["ExoMol"]]

nu_grid, wav, resolution = wavenumber_grid(
    22920.0, 23000.0, 3500, unit="AA", xsmode="premodit"
)
mul = MultiMol(molmulti=mols, dbmulti=db, database_root_path="./")
multimdb = mul.multimdb(nu_grid, Ttyp=1000.0)
snap = multimdb.to_snapshot()
