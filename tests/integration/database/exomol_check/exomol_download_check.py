# This is a small code to check Exomol download

from exojax.database.exomol.api import MdbExomol 
from exojax.utils.grids import wavenumber_grid
nus,wav,r = wavenumber_grid(22900.0,24000.0,1000,unit="AA",xsmode='premodit')
#emf='CO/12C-16O/Li2015'   #lifetime=0, Lande=0
#emf="NaH/23Na-1H/Rivlin/" #lifetime=1, Lande=0
emf="MgH/24Mg-1H/XAB/" #lifetime=1, Lande=1
mdb =MdbExomol(emf,nus,inherit_dataframe=True)
