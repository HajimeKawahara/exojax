import numpy
from exojax.spec.rtransfer import nugrid
from exojax.spec import AutoXS
#nus,wav,res=nugrid(1900.0,2300.0,200000,"cm-1")
nus=numpy.linspace(1900.0,2300.0,200000,dtype=numpy.float64) #wavenumber (cm-1)
autoxs=AutoXS(nus,"ExoMol","CO") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)

import matplotlib.pyplot as plt
plt.plot(nus,xsv)
plt.yscale("log")
plt.show()
