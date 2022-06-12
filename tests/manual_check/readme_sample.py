import matplotlib.pyplot as plt

import numpy 
from exojax.spec import voigt
nu=numpy.linspace(-10,10,100)
v=voigt(nu,1.0,2.0) 

from exojax.spec import AutoXS
nus=numpy.linspace(1900.0,2300.0,200000,dtype=numpy.float64) #wavenumber (cm-1)
autoxs=AutoXS(nus,"ExoMol","CO") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
xsv=autoxs.xsection(1000.0,1.0)

plt.plot(nus,xsv)
plt.show()
plt.clf()

ls=autoxs.linest(1000.0) #line strength for T=1000K
plt.plot(autoxs.mdb.nu_lines,ls,".")
plt.show()

from exojax.spec.rtransfer import nugrid
from exojax.spec import AutoRT
nus,wav,res=nugrid(1900.0,2300.0,200000,"cm-1")
Parr=numpy.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02    
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addcia("H2-He",0.74,0.25)       #CIA, mmr(He)=0.25
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F=autort.rtrun()

nusobs=numpy.linspace(1900.0,2300.0,10000,dtype=numpy.float64) #observation wavenumber bin (cm-1)
Fx=autort.spectrum(nusobs,100000.0,20.0,0.0) #R=100000, vsini=10km/s, RV=0km/s

plt.plot(nus,F)
plt.plot(nus,Fx)
plt.show()
