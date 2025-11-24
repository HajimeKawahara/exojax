
from exojax.database.hitemp.api import MdbHitemp, MdbExomol

mdb = MdbExomol("CO/12C-16O/Li2015", nurange=[4000.0, 4100.0], engine="vaex")
mdb = MdbExomol("CO/12C-16O/Li2015", nurange=[4000.0, 4100.0], engine="pytables")
Mdb_reduced = MdbHitemp("CO", nurange=[4000.0, 4100.0], isotope=1, elower_max=3300.0, engine="vaex")
Mdb_reduced = MdbHitemp("CO", nurange=[4000.0, 4100.0], isotope=1, elower_max=3300.0, engine="pytables")
