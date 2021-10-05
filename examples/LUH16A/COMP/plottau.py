import matplotlib.pyplot as plt
import numpy as np

nus_lpf,dtauCO_lpf,dtauH2O_lpf=np.load("dtau_lpf.npz",allow_pickle=True)["arr_0"]
nus_modit,dtauCO_modit,dtauH2O_modit=np.load("dtau_modit.npz",allow_pickle=True)["arr_0"]

from exojax.plot.atmplot import plottau
from exojax.spec import rtransfer as rt
#ATMOSPHERE                                                                                                             
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr=(Parr)**0.01


i=50
plt.plot(nus_lpf,dtauCO_lpf[i,:])
plt.plot(nus_modit,dtauCO_modit[i,:])

i=10
plt.plot(nus_lpf,dtauCO_lpf[i,:])
plt.plot(nus_modit,dtauCO_modit[i,:])

plt.title(str(Parr[i]))
plt.yscale("log")
plt.show()
