import numpy
from exojax.spec import AutoXS
import pytest
import jax.numpy as jnp

nus=numpy.linspace(1900.0,2300.0,40000,dtype=numpy.float64)
nuslog=numpy.logspace(numpy.log10(1900.0),numpy.log10(2300.0),40000,dtype=numpy.float64)

autoxs=AutoXS(nuslog,"ExoMol","CO",xsmode="REDIT") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv0=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)
autoxs=AutoXS(nuslog,"ExoMol","CO",xsmode="LPF") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv1=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)
dif=(numpy.sum((xsv0-xsv1)**2))

print("difference")
print("REDIT-LPF:",dif)

import matplotlib.pyplot as plt
plt.plot(nus,xsv0,label="REDIT")
plt.plot(nus,xsv1,".",label="LPF")
plt.legend()
plt.show()


autoxs=AutoXS(nus,"ExoMol","CO",xsmode="DIT") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv0=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)

autoxs=AutoXS(nus,"ExoMol","CO",xsmode="LPF") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv1=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)
dif=(numpy.sum((xsv0-xsv1)**2))

import matplotlib.pyplot as plt
plt.plot(nus,xsv0,label="DIT")
plt.plot(nus,xsv1,".",label="LPF")
plt.legend()
plt.show()

print("difference")
print("DIT-LPF:",dif)

autoxs=AutoXS(nuslog,"ExoMol","CO",xsmode="MODIT") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv0=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)

autoxs=AutoXS(nuslog,"ExoMol","CO",xsmode="LPF") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv1=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)
dif=(numpy.sum((xsv0-xsv1)**2))

print("difference")
print("MODIT-LPF:",dif)

import matplotlib.pyplot as plt
plt.plot(nus,xsv0,label="MODIT")
plt.plot(nus,xsv1,".",label="LPF")
plt.legend()
plt.show()
