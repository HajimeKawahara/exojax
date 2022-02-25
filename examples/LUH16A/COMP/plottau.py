import matplotlib.pyplot as plt
import numpy as np

nus_lpf,dtauCO_lpf,dtauH2O_lpf=np.load("dtau_lpf.npz",allow_pickle=True)["arr_0"]
nus_modit,dtauCO_modit,dtauH2O_modit=np.load("dtau_modit4500.npz",allow_pickle=True)["arr_0"]
#nus_modit64,dtauCO_modit64,dtauH2O_modit64=np.load("dtau_modit64.npz",allow_pickle=True)["arr_0"]
#nus_modit_high,dtauCO_modit_high,dtauH2O_modit_high=np.load("dtau_modit9000.npz",allow_pickle=True)["arr_0"]

from exojax.plot.atmplot import plottau
from exojax.spec import rtransfer as rt

#ATMOSPHERE                                                                                                            
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr=(Parr)**0.01

fig=plt.figure(figsize=(8,4))
i=50
plt.plot(nus_modit,dtauCO_modit[i,:],color="C1",label="MODIT, P="+str(Parr[i])+" bar")
plt.plot(nus_lpf,dtauCO_lpf[i,:],color="C0",label="LPF, P="+str(Parr[i])+" bar",ls="dashed")
#plt.plot(nus_modit64,dtauCO_modit64[i,:])
#plt.plot(nus_modit_high,dtauCO_modit_high[i,:],color="C2",label="MODIT (high), P="+str(Parr[i])+" bar")

i=80
plt.plot(nus_modit,dtauCO_modit[i,:],color="C3",label="MODIT, P="+str(Parr[i])+" bar")
plt.plot(nus_lpf,dtauCO_lpf[i,:],color="C2",label="LPF, P="+str(Parr[i])+" bar",ls="dashed")
#plt.plot(nus_modit_high,dtauCO_modit_high[i,:],lw=0.5,color="C2",label="MODIT (high), P="+str(Parr[i])+" bar")
#plt.plot(nus_modit64,dtauCO_modit64[i,:])
plt.xlabel("wavenumber (cm-1)")
plt.ylabel("delta tau")

plt.legend()
plt.yscale("log")
plt.savefig("comp_luhman16A.png")
plt.show()
