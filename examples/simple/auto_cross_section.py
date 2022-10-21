import matplotlib.pyplot as plt
import numpy
from exojax.utils.grids import wavenumber_grid
from exojax.spec import AutoXS
# nus,wav,res=nugrid(1900.0,2300.0,200000,"cm-1")
nus = numpy.linspace(1900.0, 2300.0, 200000,
                     dtype=numpy.float64)  # wavenumber (cm-1)
# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nus, 'ExoMol', 'CO')
xsv = autoxs.xsection(1000.0, 1.0)  # cross section for 1000K, 1bar (cm2)

plt.plot(nus, xsv)
plt.yscale('log')
plt.show()
