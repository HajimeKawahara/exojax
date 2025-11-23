from exojax.database.exomol.api import MdbExomol
from exojax.utils.grids import wavenumber_grid
nu_grid, wav, res = wavenumber_grid(22900,23000,10000, xsmode="premodit", unit="AA")
mdbH2S = MdbExomol(".database/H2S/1H2-32S/AYT2", nurange=nu_grid, gpu_transfer=False)
#mdbCO = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid, gpu_transfer=False)