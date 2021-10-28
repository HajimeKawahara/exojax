import numpy as np
from exojax.spec.rtransfer import nugrid
from exojax.spec import AutoRT
import matplotlib.pyplot as plt

xsmode="MODIT"
nus,wav,res=nugrid(1900.0,2300.0,160000,"cm-1",xsmode=xsmode)
Parr=np.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02    

autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F0=autort.rtrun()

xsmode="LPF"
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F1=autort.rtrun()
print("MODIT and LPF DIF=",np.sum((F0-F1)**2)/len(F0)/np.median(F0))


plt.plot(nus,F0,label="MODIT")
#plt.plot(nus,F1,".",label="LPF")
plt.legend()
plt.show()


xsmode="DIT"
nus,wav,res=nugrid(1900.0,2300.0,160000,"cm-1",xsmode=xsmode)
Parr=np.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02

autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F0=autort.rtrun()

xsmode="LPF"
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F1=autort.rtrun()
print("DIT and LPF DIF=",np.sum((F0-F1)**2)/len(F0)/np.median(F0))

plt.plot(nus,F0)
plt.plot(nus,F1,".")
plt.show()


