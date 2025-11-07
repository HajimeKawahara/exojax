#
# download cia database and test OpaCIA
#
from exojax.utils.grids import wavenumber_grid
from exojax.database.cia.api import CdbCIA
from exojax.opacity import OpaCIA

nu_grid, wav, res = wavenumber_grid(
    23000.0, 23100.0, 1500, unit="AA", xsmode="premodit"
)

cdbH2H2 = CdbCIA(".database/H2-H2_2011.cia", nu_grid)
opacia = OpaCIA(cdbH2H2, nu_grid)
