import matplotlib.pyplot as plt
import numpy
from exojax.utils.grids import wavenumber_grid
from exojax.spec import AutoRT
nus, wav, res = wavenumber_grid(1900.0, 2300.0, 200000, 'cm-1')
Parr = numpy.logspace(-8, 2, 100)  # 100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02
autort = AutoRT(nus, 1.e5, 2.33, Tarr, Parr)  # g=1.e5 cm/s2, mmw=2.33
autort.addcia('H2-H2', 0.74, 0.74)  # CIA, mmr(H)=0.74
autort.addcia('H2-He', 0.74, 0.25)  # CIA, mmr(He)=0.25
autort.addmol('ExoMol', 'CO', 0.01)  # CO line, mmr(CO)=0.01
F = autort.rtrun()

plt.plot(nus, F)
plt.show()
